# my_attn.py

import torch
from block_sparse_attn import block_sparse_attn_func
from .vis_utils import visualize_block_mask

IS_DEBUG = False
IS_TIMING = False

class AttentionCache:
    """Attention 相关 tensor 的缓存，避免重复创建"""

    def __init__(self):
        self._cu_seqlens = {}
        self._head_mask_type = {}

    def get_cu_seqlens(self, seq_len, device):
        key = (seq_len, str(device))
        if key not in self._cu_seqlens:
            self._cu_seqlens[key] = torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            )
        return self._cu_seqlens[key]

    def get_head_mask_type(self, num_heads, device):
        key = (num_heads, str(device))
        if key not in self._head_mask_type:
            self._head_mask_type[key] = torch.tensor(
                [1] * num_heads, dtype=torch.int32, device=device
            )
        return self._head_mask_type[key]


# 全局缓存实例
_attention_cache = AttentionCache()



def create_deterministic_block_mask(
        q_block_num: int,
        k_block_num: int,
        H: int,
        device: torch.device,
        keep_sink: int = 4,
        keep_recent: int = 4,
        local_window: int = 8,
) -> torch.Tensor:
    """
    创建确定性的 block mask，仅包含 sink + recent + local 策略。

    Returns:
        mask: [1, H, q_block_num, k_block_num] (bool)
    """


    # 2. Sink & recent
    idx_k = torch.arange(k_block_num, device=device)
    sink_mask = idx_k < keep_sink
    recent_mask = idx_k >= (k_block_num - keep_recent)
    sink_mask = sink_mask[None, None, :]   # [1, 1, k_b]
    recent_mask = recent_mask[None, None, :]

    # 3. Local window
    if local_window > 0:
        idx_q = torch.arange(q_block_num, device=device)[:, None]
        idx_k_mat = idx_k[None, :]
        local_mask = (idx_q - idx_k_mat).abs() <= local_window
        local_mask = local_mask[None, :, :]
    else:
        local_mask = torch.zeros(1, q_block_num, k_block_num, dtype=torch.bool, device=device)

    # 4. 合并
    final_mask = (
        sink_mask.expand(H, q_block_num, k_block_num) |
        recent_mask.expand(H, q_block_num, k_block_num) |
        local_mask.expand(H, q_block_num, k_block_num)
    )

    return final_mask.unsqueeze(0)  # [1, H, q_b, k_b]


def myattn_prefill(
        query_states,
        key_states,
        value_states,
        layer_idx,
        block_size=128,
        is_causal=True,
        is_visual=True,
        sink_ratio=0.1,
        recent_ratio=0.1,
        local_span_ratio=0.1,
):
    """
    执行自定义 Block-Sparse Attention（仅用于 prefilling 阶段）
    仅支持 CUDA。
    若 IS_TIMING=True，则对各阶段进行精确耗时分析。
    """
    assert query_states.is_cuda, "输入张量必须位于 CUDA 上"
    bsz = 1
    L_q = query_states.shape[2]  # 查询序列长度
    L_k = key_states.shape[2]  # 键/值序列长度
    H = query_states.shape[1]  # 注意力头数
    head_dim = query_states.shape[3]  # 每个头的维度
    device = query_states.device

    # --- 可选：初始化 timing events ---
    if IS_TIMING:
        events = {
            "start": torch.cuda.Event(enable_timing=True),
            "after_reshape": torch.cuda.Event(enable_timing=True),
            "after_mask": torch.cuda.Event(enable_timing=True),
            "before_cache": torch.cuda.Event(enable_timing=True),
            "after_cache": torch.cuda.Event(enable_timing=True),
            "after_kernel": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        torch.cuda.synchronize()
        events["start"].record()
    else:
        events = None

    # 📌 阶段 1: 将张量重塑为 [L, H, D] 格式（去除 batch 维度并转置）
    q_unpad = query_states.squeeze(0).transpose(0, 1).contiguous()  # [L_q, H, D]
    k_unpad = key_states.squeeze(0).transpose(0, 1).contiguous()  # [L_k, H, D]
    v_unpad = value_states.squeeze(0).transpose(0, 1).contiguous()  # [L_k, H, D]

    if IS_TIMING:
        torch.cuda.synchronize()
        events["after_reshape"].record()

    # 📌 阶段 2: 计算 block 数量（向上取整）
    q_block_num = (L_q + block_size - 1) // block_size
    k_block_num = (L_k + block_size - 1) // block_size

    if q_block_num > 1 and IS_DEBUG:
        print(f"查询长度 L_q: {L_q}, 键长度 L_k: {L_k}, 头数 H: {H}, "
              f"查询 block 数: {q_block_num}, 键 block 数: {k_block_num}")

    # 📌 阶段 3: 构建确定性稀疏 block 掩码（sink + recent + local window）
    keep_sink = max(1, int(k_block_num * sink_ratio))
    keep_recent = max(1, int(k_block_num * recent_ratio))
    local_span = max(2, int(k_block_num * local_span_ratio))
    local_window = local_span // 2

    base_blockmask = create_deterministic_block_mask(
        q_block_num=q_block_num,
        k_block_num=k_block_num,
        H=H,
        device=device,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        local_window=local_window,
    ).contiguous()

    if layer_idx == 0:
        print("使用确定性稀疏 block 掩码（sink + recent + 局部窗口）...")

    # 可视化掩码（仅在长序列且启用可视化时）
    if q_unpad.shape[0] > 200 and is_visual:
        total = base_blockmask.numel()
        kept = base_blockmask.sum().item()
        ratio = kept / total
        print(f"  => 第 {layer_idx} 层: 确定性 block 掩码保留率: {ratio:.4f} ({kept}/{total})")
        visualize_block_mask(
            base_blockmask,
            layer_idx=layer_idx,
            seq_len=L_q,
            keep_ratio=ratio,
            head_idx=0,
            threshold=None,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            local_window=local_window,
            save_dir="./vis_masks"
        )

    if IS_TIMING:
        torch.cuda.synchronize()
        events["after_mask"].record()

    # 📌 阶段 4: 获取缓存的 tensor（几乎零开销）
    if IS_TIMING:
        torch.cuda.synchronize()
        events["before_cache"].record()

    q_cu_seqlens = _attention_cache.get_cu_seqlens(L_q, device)
    k_cu_seqlens = _attention_cache.get_cu_seqlens(L_k, device)
    head_mask_type = _attention_cache.get_head_mask_type(H, device)

    if IS_TIMING:
        torch.cuda.synchronize()
        events["after_cache"].record()

    # 📌 阶段 5: 调用底层 kernel
    attn_output_unpad = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        q_cu_seqlens, k_cu_seqlens,
        head_mask_type=head_mask_type,
        streaming_info=None,
        base_blockmask=base_blockmask,
        max_seqlen_q_=L_q,
        max_seqlen_k_=L_k,
        p_dropout=0.0,
        deterministic=True,
        is_causal=is_causal,
    )  # [L_q, H, D]

    if IS_TIMING:
        torch.cuda.synchronize()
        events["after_kernel"].record()

    # 📌 阶段 6: 恢复输出格式
    attn_output = attn_output_unpad.transpose(0, 1).unsqueeze(0)  # [1, H, L_q, D]
    attn_output = attn_output.transpose(1, 2).reshape(bsz, L_q, -1)  # [1, L_q, H*D]

    if IS_TIMING:
        torch.cuda.synchronize()
        events["end"].record()

    # --- 可选：同步并打印耗时 ---
    if IS_TIMING:
        torch.cuda.synchronize()

        def ms(a, b):
            return a.elapsed_time(b)

        if q_unpad.shape[0] > 2000:
            t_reshape = ms(events["start"], events["after_reshape"])
            t_mask = ms(events["after_reshape"], events["after_mask"])
            t_cache = ms(events["after_mask"], events["after_cache"])
            t_cache_real = ms(events["before_cache"], events["after_cache"])
            t_kernel = ms(events["after_cache"], events["after_kernel"])
            t_reshape_back = ms(events["after_kernel"], events["end"])

            total = t_reshape + t_mask + t_cache + t_kernel + t_reshape_back
            print(f"[第 {layer_idx:2d} 层] "
                  f"重塑: {t_reshape:6.2f}ms | "
                  f"掩码: {t_mask:6.2f}ms | "
                  f"缓存: {t_cache:6.2f}ms | "
                  f"缓存_real: {t_cache_real:6.2f}ms | "
                  f"Kernel: {t_kernel:6.2f}ms | "
                  f"恢复: {t_reshape_back:6.2f}ms | "
                  f"总计: {total:6.2f}ms")

    return attn_output
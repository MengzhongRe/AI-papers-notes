# Byte-to-Unicode 映射工具 — GPT-2 经典设计
#
# 将 0–255 的字节值映射到可见 Unicode 字符，
# 避免空格、换行符等控制字符干扰正则表达式处理。
#
# 使用方式：
#   from byte_to_unicode_map import bytes_to_unicode


def bytes_to_unicode():
    """创建 0–255 字节到可见 Unicode 字符的映射表。

    OpenAI GPT-2 引入的技巧：将原始字节流中的不可见控制字符
    (如空格 0x20、换行 0x0A) 映射到特殊的可见 Unicode 字符，
    使它们能安全地通过正则引擎处理，且方便调试。

    Returns:
        dict: 字节值 (int) → 可见 Unicode 字符 (str)
    """
    # 优先使用 ASCII 可见字符 + Latin-1 扩展区
    bs = (
        list(range(ord("!"), ord("~") + 1))   # ! 到 ~ (33–126)
        + list(range(ord("¡"), ord("¬") + 1))  # ¡ 到 ¬ (161–172)
        + list(range(ord("®"), ord("ÿ") + 1))  # ® 到 ÿ (174–255)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

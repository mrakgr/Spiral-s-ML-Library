def find_nth_set_bit(mask, base, offset):
    # Consider bits starting from 'base'
    masked_bits = mask >> base
    count = 0
    for i in range(32 - base):
        if masked_bits & (1 << i):
            count += 1
            if count == offset:
                return base + i
    return 0xFFFFFFFF  # Not found

find_nth_set_bit(16,0,1)
namespace rt
{
void BVHTraverse4Test()
{
    for (f32 a = 0; a < 4; a++)
    {
        for (f32 b = 0; b < 4; b++)
        {
            for (f32 c = 0; c < 4; c++)
            {
                for (f32 d = 0; d < 4; d++)
                {
                    alignas(16) f32 values[4] = {a, b, c, d};
                    u32 order[4]              = {0, 1, 2, 3};
                    for (u32 sort0 = 1; sort0 < 4; sort0++)
                    {
                        f32 key   = values[sort0];
                        i32 sort1 = sort0 - 1;
                        while (sort1 >= 0 && values[sort1] > key)
                        {
                            values[sort1 + 1] = values[sort1];
                            order[sort1 + 1]  = order[sort1];
                            sort1--;
                        }
                        values[sort1 + 1] = key;
                        order[sort1 + 1]  = sort0;
                    }
                    u32 outOrder[4];
                    for (u32 i = 0; i < 4; i++)
                    {
                        outOrder[order[i]] = i;
                    }

                    Lane4F32 t_dcba(a, b, c, d);
                    const Lane4F32 abac = ShuffleReverse<0, 1, 0, 2>(t_dcba);
                    const Lane4F32 adcd = ShuffleReverse<0, 3, 2, 3>(t_dcba);

                    const u32 da_cb_ba_ac = Movemask(t_dcba < abac) & 0xe;
                    const u32 aa_db_ca_dc = Movemask(adcd < abac);

                    u32 da_cb_ba_db_ca_dc = da_cb_ba_ac * 4 + aa_db_ca_dc;

                    u32 indexA = PopCount(da_cb_ba_db_ca_dc & 0x2a);
                    u32 indexB = PopCount((da_cb_ba_db_ca_dc ^ 0x08) & 0x1c);
                    u32 indexC = PopCount((da_cb_ba_db_ca_dc ^ 0x12) & 0x13);
                    u32 indexD = PopCount((~da_cb_ba_db_ca_dc) & 0x25);

                    Assert(indexA == outOrder[0]);
                    Assert(indexB == outOrder[1]);
                    Assert(indexC == outOrder[2]);
                    Assert(indexD == outOrder[3]);
                }
            }
        }
    }
}

void BVHTraverse8Test()
{
    f32 time      = 0.f;
    f32 time2     = 0.f;
    auto testCase = [&](Lane8F32 t_hgfedcba, u32 order[8]) {
        PerformanceCounter counter = OS_StartCounter();
        Lane8F32 t_aaaaaaaa        = Shuffle<0>(t_hgfedcba);
        Lane8F32 t_edbcbbca        = ShuffleReverse<4, 3, 1, 2, 1, 1, 2, 0>(t_hgfedcba);
        Lane8F32 t_gfcfeddb        = ShuffleReverse<6, 5, 2, 5, 4, 3, 3, 1>(t_hgfedcba);
        Lane8F32 t_hhhgfgeh        = ShuffleReverse<7, 7, 7, 6, 5, 6, 4, 7>(t_hgfedcba);

        const u32 mask0 = Movemask(t_aaaaaaaa < t_gfcfeddb);
        const u32 mask1 = Movemask(t_edbcbbca < t_gfcfeddb);
        const u32 mask2 = Movemask(t_edbcbbca < t_hhhgfgeh);
        const u32 mask3 = Movemask(t_gfcfeddb < t_hhhgfgeh);

        const u32 mask = mask0 | (mask1 << 8) | (mask2 << 16) | (mask3 << 24);

        u32 indexA = PopCount(~mask & 0x000100ed);
        u32 indexB = PopCount((mask ^ 0x002c2c00) & 0x002c2d00);
        u32 indexC = PopCount((mask ^ 0x20121200) & 0x20123220);
        u32 indexD = PopCount((mask ^ 0x06404000) & 0x06404602);
        u32 indexE = PopCount((mask ^ 0x08808000) & 0x0a828808);
        u32 indexF = PopCount((mask ^ 0x50000000) & 0x58085010);
        u32 indexG = PopCount((mask ^ 0x80000000) & 0x94148080);
        u32 indexH = PopCount(mask & 0xe0e10000);
        Assert(indexA == order[0]);
        Assert(indexB == order[1]);
        Assert(indexC == order[2]);
        Assert(indexD == order[3]);
        Assert(indexE == order[4]);
        Assert(indexF == order[5]);
        Assert(indexG == order[6]);
        Assert(indexH == order[7]);
        time += OS_GetMilliseconds(counter);
    };
    for (u32 a = 0; a < 8; a++)
    {
        for (u32 b = 0; b < 8; b++)
        {
            for (u32 c = 0; c < 8; c++)
            {
                for (u32 d = 0; d < 8; d++)
                {
                    for (u32 e = 0; e < 8; e++)
                    {
                        for (u32 f = 0; f < 8; f++)
                        {
                            for (u32 g = 0; g < 8; g++)
                            {
                                for (u32 h = 0; h < 8; h++)
                                {
                                    PerformanceCounter counter = OS_StartCounter();
                                    alignas(32) u32 values[8]  = {a, b, c, d, e, f, g, h};
                                    alignas(32) u32 out[8]     = {a, b, c, d, e, f, g, h};
                                    u32 order[8]               = {0, 1, 2, 3, 4, 5, 6, 7};
                                    for (u32 sort0 = 1; sort0 < 8; sort0++)
                                    {
                                        u32 key   = values[sort0];
                                        i32 sort1 = sort0 - 1;
                                        while (sort1 >= 0 && values[sort1] >= key)
                                        {
                                            values[sort1 + 1] = values[sort1];
                                            order[sort1 + 1]  = order[sort1];
                                            sort1--;
                                        }
                                        values[sort1 + 1] = key;
                                        order[sort1 + 1]  = sort0;
                                    }
                                    time2 += OS_GetMilliseconds(counter);
                                    u32 outOrder[8];
                                    for (u32 i = 0; i < 8; i++)
                                    {
                                        outOrder[order[i]] = i;
                                    }

                                    testCase(Lane8F32(Lane8U32::Load(out)), outOrder);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // printf("time %fms\n", OS_GetMilliseconds(counter));
    printf("time avx %fms\n", time);
    printf("time insertion %fms\n", time2);
}

} // namespace rt

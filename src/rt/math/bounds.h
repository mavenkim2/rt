#ifndef BOUNDS_H
#define BOUNDS_H

namespace rt
{
struct Bounds
{
    Lane4F32 minP;
    Lane4F32 maxP;

    Bounds() : minP(pos_inf), maxP(neg_inf) {}
    Bounds(Vec3f minP, Vec3f maxP) : minP(Lane4F32(minP)), maxP(Lane4F32(maxP)) {}
    Bounds(const Lane4F32 &minP, const Lane4F32 &maxP) : minP(minP), maxP(maxP) {}
    Bounds(const Lane8F32 &l) : minP(-Extract4<0>(l)), maxP(Extract4<1>(l)) {}

    __forceinline bool Empty() const { return (Movemask(minP > maxP) & 0x7) != 0; }

    __forceinline void Extend(Lane4F32 inMin, Lane4F32 inMax)
    {
        minP = Min(minP, inMin);
        maxP = Max(maxP, inMax);
    }
    __forceinline void Extend(Lane4F32 in)
    {
        minP = Min(minP, in);
        maxP = Max(maxP, in);
    }
    __forceinline void Extend(Vec3f in) { return Extend(Lane4F32(in)); }
    __forceinline void Extend(const Bounds &other)
    {
        minP = Min(minP, other.minP);
        maxP = Max(maxP, other.maxP);
    }
    __forceinline bool Contains(const Bounds &other) const
    {
        return All(other.minP >= minP) && All(other.maxP <= maxP);
    }
    __forceinline void Intersect(const Bounds &other)
    {
        minP = Max(other.minP, minP);
        maxP = Min(other.maxP, maxP);
    }
    __forceinline Lane4F32 Diagonal() const { return maxP - minP; }
    __forceinline Lane4F32 Centroid() const { return (maxP + minP) * 0.5f; }
    __forceinline Lane4F32 operator[](u32 index)
    {
        Assert(index == 0 || index == 1);
        return index ? maxP : minP;
    }
};

__forceinline Bounds Intersect(const Bounds &a, const Bounds &b)
{
    Bounds result;
    result.minP = Max(a.minP, b.minP);
    result.maxP = Min(a.maxP, b.maxP);
    return result;
}
__forceinline bool Intersects(const Bounds &a, const Bounds &b)
{
    Bounds test = Intersect(a, b);
    return None(test.minP > test.maxP);
}

__forceinline f32 HalfArea(const Bounds &b)
{
    Lane4F32 extent = b.maxP - b.minP;
    return FMA(extent[0], extent[1] + extent[2], extent[1] * extent[2]);
}

static const Lane8F32 signFlipMask(-0.f, -0.f, -0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
static const u32 LUTAxis[] = {1, 2, 0};

struct Bounds8
{
    Lane8F32 v;

    Bounds8() : v(neg_inf) {}
    Bounds8(EmptyTy) : v(neg_inf) {}
    Bounds8(PosInfTy) : v(pos_inf) {}
    __forceinline explicit Bounds8(Lane8F32 in) { v = in ^ signFlipMask; }

    __forceinline explicit operator const __m256 &() const { return v; }
    __forceinline explicit operator __m256 &() { return v; }

    __forceinline explicit operator Bounds() const
    {
        Bounds out;
        out.minP = FlipSign(Extract4<0>(v));
        out.maxP = Extract4<1>(v);
        return out;
    }

    __forceinline void Extend(const Lane8F32 &other) { v = Max(v, other); }
    __forceinline void Extend(const Bounds8 &other) { v = Max(v, other.v); }
    __forceinline void MaskExtend(const Lane8F32 &mask, const Bounds8 &other)
    {
        v = MaskMax(mask, v, other.v);
    }
    __forceinline void Intersect(const Bounds8 &other) { v = Min(v, other.v); }
    __forceinline bool Empty() const
    {
        return (Movemask(-Extract4<0>(v) > Extract4<1>(v)) & 0x77) != 0;
    }
};

__forceinline Bounds8 Intersect(const Bounds8 &l, const Bounds8 &r)
{
    Bounds8 out;
    out.v = Min(l.v, r.v);
    return out;
}

__forceinline f32 HalfArea(const Bounds8 &b)
{
    Lane4F32 mins   = Extract4<0>(b.v);
    Lane4F32 maxs   = Extract4<1>(b.v);
    Lane4F32 extent = (maxs + mins);
    return FMA(extent[0], extent[1] + extent[2], extent[1] * extent[2]);
}

__forceinline f32 HalfArea(const Lane8F32 &b)
{
    Lane4F32 mins   = Extract4<0>(b);
    Lane4F32 maxs   = Extract4<1>(b);
    Lane4F32 extent = (maxs + mins);
    return FMA(extent[0], extent[1] + extent[2], extent[1] * extent[2]);
}

struct Bounds8F32
{
    Lane8F32 minU;
    Lane8F32 minV;
    Lane8F32 minW;

    Lane8F32 maxU;
    Lane8F32 maxV;
    Lane8F32 maxW;

    Bounds8F32()
        : minU(pos_inf), minV(pos_inf), minW(pos_inf), maxU(neg_inf), maxV(neg_inf),
          maxW(neg_inf)
    {
    }
    Bounds8F32(NegInfTy)
        : minU(neg_inf), minV(neg_inf), minW(neg_inf), maxU(neg_inf), maxV(neg_inf),
          maxW(neg_inf)
    {
    }

    __forceinline Bounds ToBounds()
    {
        f32 sMinU = ReduceMin(minU);
        f32 sMinV = ReduceMin(minV);
        f32 sMinW = ReduceMin(minW);

        f32 sMaxU = ReduceMax(maxU);
        f32 sMaxV = ReduceMax(maxV);
        f32 sMaxW = ReduceMax(maxW);

        return Bounds(Lane4F32(sMinU, sMinV, sMinW, 0.f), Lane4F32(sMaxU, sMaxV, sMaxW, 0.f));
    }

    __forceinline Bounds ToBoundsNegMin()
    {
        f32 sMinU = ReduceMax(minU);
        f32 sMinV = ReduceMax(minV);
        f32 sMinW = ReduceMax(minW);

        f32 sMaxU = ReduceMax(maxU);
        f32 sMaxV = ReduceMax(maxV);
        f32 sMaxW = ReduceMax(maxW);

        return Bounds(Lane4F32(-sMinU, -sMinV, -sMinW, 0.f),
                      Lane4F32(sMaxU, sMaxV, sMaxW, 0.f));
    }
    __forceinline void Extend(const Bounds8F32 &other)
    {
        minU = Min(minU, other.minU);
        minV = Min(minV, other.minV);
        minW = Min(minW, other.minW);

        maxU = Max(maxU, other.maxU);
        maxV = Max(maxV, other.maxV);
        maxW = Max(maxW, other.maxW);
    }
    __forceinline void Extend(const Lane8F32 &x, const Lane8F32 &y, const Lane8F32 &z)
    {
        minU = Min(minU, x);
        minV = Min(minV, y);
        minW = Min(minW, z);

        maxU = Max(maxU, x);
        maxV = Max(maxV, y);
        maxW = Max(maxW, z);
    }
    __forceinline void MaskExtend(const Lane8F32 &mask, const Lane8F32 &x, const Lane8F32 &y,
                                  const Lane8F32 &z)
    {
        minU = MaskMin(mask, minU, x);
        minV = MaskMin(mask, minV, y);
        minW = MaskMin(mask, minW, z);

        maxU = MaskMax(mask, maxU, x);
        maxV = MaskMax(mask, maxV, y);
        maxW = MaskMax(mask, maxW, z);
    }
    __forceinline void MaskExtend(const Lane8F32 &mask, const Lane8F32 &minX,
                                  const Lane8F32 &minY, const Lane8F32 &minZ,
                                  const Lane8F32 &maxX, const Lane8F32 &maxY,
                                  const Lane8F32 &maxZ)
    {
        minU = MaskMin(mask, minU, minX);
        minV = MaskMin(mask, minV, minY);
        minW = MaskMin(mask, minW, minZ);

        maxU = MaskMax(mask, maxU, maxX);
        maxV = MaskMax(mask, maxV, maxY);
        maxW = MaskMax(mask, maxW, maxZ);
    }
    __forceinline void MaskExtendNegMin(const Lane8F32 &mask, const Lane8F32 &x,
                                        const Lane8F32 &y, const Lane8F32 &z)
    {
        minU = MaskMax(mask, minU, x);
        minV = MaskMax(mask, minV, y);
        minW = MaskMax(mask, minW, z);

        maxU = MaskMax(mask, maxU, x);
        maxV = MaskMax(mask, maxV, y);
        maxW = MaskMax(mask, maxW, z);
    }
    __forceinline void MaskExtendNegMin(const Lane8F32 &mask, const Lane8F32 &minX,
                                        const Lane8F32 &minY, const Lane8F32 &minZ,
                                        const Lane8F32 &maxX, const Lane8F32 &maxY,
                                        const Lane8F32 &maxZ)
    {
        minU = MaskMax(mask, minU, minX);
        minV = MaskMax(mask, minV, minY);
        minW = MaskMax(mask, minW, minZ);

        maxU = MaskMax(mask, maxU, maxX);
        maxV = MaskMax(mask, maxV, maxY);
        maxW = MaskMax(mask, maxW, maxZ);
    }
    __forceinline void ExtendNegMin(const Lane8F32 &minX, const Lane8F32 &minY,
                                    const Lane8F32 &minZ, const Lane8F32 &maxX,
                                    const Lane8F32 &maxY, const Lane8F32 &maxZ)
    {
        minU = Max(minU, minX);
        minV = Max(minV, minY);
        minW = Max(minW, minZ);

        maxU = Max(maxU, maxX);
        maxV = Max(maxV, maxY);
        maxW = Max(maxW, maxZ);
    }

    __forceinline void MaskExtendL(const Lane8F32 &mask, const Lane8F32 &u, const Lane8F32 &v,
                                   const Lane8F32 &w)
    {
        minU = MaskMin(mask, minU, u);
        minV = MaskMin(mask, minV, v);
        minW = MaskMin(mask, minW, w);

        maxV = MaskMax(mask, maxV, v);
        maxW = MaskMax(mask, maxW, w);
    }

    __forceinline void MaskExtendR(const Lane8F32 &mask, const Lane8F32 &u, const Lane8F32 &v,
                                   const Lane8F32 &w)
    {

#if 1
        minV = MaskMin(mask, minV, v);
        minW = MaskMin(mask, minW, w);

        maxU = MaskMax(mask, maxU, u);
        maxV = MaskMax(mask, maxV, v);
        maxW = MaskMax(mask, maxW, w);
#else
        minV = Select(mask, minV, Min(minV, v));
        minW = Select(mask, minW, Min(minW, w));

        maxU = Select(mask, maxU, Max(maxU, u));
        maxV = Select(mask, maxV, Max(maxV, v));
        maxW = Select(mask, maxW, Max(maxW, w));
#endif
    }

    __forceinline void MaskExtendVW(const Lane8F32 &mask, const Lane8F32 &v, const Lane8F32 &w)
    {
        minV = MaskMin(mask, minV, v);
        maxV = MaskMax(mask, maxV, v);

        minW = MaskMin(mask, minW, w);
        maxW = MaskMax(mask, maxW, w);
    }
};

} // namespace rt
#endif

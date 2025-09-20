#ifndef DISNEY_BSDF_HLSLI
#define DISNEY_BSDF_HLSLI

#include "../sampling.hlsli"

float SchlickFresnel(float u)
{
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m;
}

#if 0
float3 DisneyThin(float wo, float3 wi) 
{
    float sheen = 0.15f;
    float sheenTint = 0.3f;
    float3 baseColor = float3(.554, .689, .374);
    float flatness = .25f;
    float roughness = 0.72f;
    float metallic = 0.f;
    float specTrans = 0.f;

    float cosThetaD;
    float FH;
    
    // Sheen
    float3 cdLin = float3(pow(baseColor.x, 2.2), pow(baseColor.y, 2.2), pow(baseColor.z, 2.2));
    float cdLum = .3 * cdLin.x + .6 * cdLin.y + .1 * cdLin.z;
    float3 cTint = cdLum > 0 ? cdLin / cdLum : 1.f;
    float3 cSheen = lerp(1.f, cTint, sheenTint);
    float3 fSheen = FH * sheen * cSheen;

    // Thin surface BSDF
    float ior = 1.22;
    float transissionRoughness = saturate(roughness * (.65f * ior - .35f));

    // Thin surface BRDF
    float fss90 = ldotH * ldotH * roughness;
    float fL = SchlickFresnel(wi.z); 
    float fV = SchlickFresnel(wo.z);

    float3 h;
    float wiDotH = Dot(wi, h);
    float fd = (1 - 0.5f * fL) * (1 - 0.5f * fV);
    float rr = 2 * roughness * wiDotH * wiDotH;
    float retroReflection = rr * (fL + fV + fL * fV * (rr - 1.f));

    float fss = lerp(1.f, fss90, fL) * lerp(1.f, fss90, fV);
    float ss = 1.25f * (fss * (1.f / (wi.z + wo.z) - .5f) + .5f);

    float3 diffuse = cdLin * InvPi * lerp(fd + retroReflection, ss, flatness);

    float3 result = diffuse * (1 - metallic)

}
#endif

float3 SampleDisneyThin(float2 u, inout float3 throughput, float3 wo) 
{
    float sheen = 0.15f;
    float sheenTint = 0.3f;
    float3 baseColor = float3(.554, .689, .374);
    float flatness = .25f;
    float roughness = 0.72f;
    float metallic = 0.f;
    float specTrans = 0.f;

    float diffTrans = 1.1f; 
    diffTrans /= 2.f;

    float probDiffReflect = (1.f - specTrans) * (1.f - diffTrans);
    float probDiffRefract = (1.f - specTrans) * diffTrans;
    float totalProb = probDiffRefract + probDiffReflect;
    probDiffReflect /= totalProb;
    probDiffRefract /= totalProb;

    float3 dir;
    float3 value;
    float3 cdLin = float3(pow(baseColor.x, 2.2), pow(baseColor.y, 2.2), pow(baseColor.z, 2.2));
    // TODO: specular BSDF
    if (u.x < probDiffReflect)
    {
        u.x = u.x / probDiffReflect;
        dir = SampleCosineHemisphere(u);

        float3 wi = dir;
        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);

        float fH = SchlickFresnel(wiDotH);
        float cdLum = .3 * cdLin.x + .6 * cdLin.y + .1 * cdLin.z;
        float3 cTint = cdLum > 0 ? cdLin / cdLum : 1.f;
        float3 cSheen = lerp(1.f, cTint, sheenTint);
        float3 fSheen = fH * sheen * cSheen;

        float fss90 = wiDotH * wiDotH * roughness;
        float fL = SchlickFresnel(wi.z); 
        float fV = SchlickFresnel(wo.z);

        float fd = (1 - 0.5f * fL) * (1 - 0.5f * fV);
        float rr = 2 * roughness * wiDotH * wiDotH;
        float retroReflection = rr * (fL + fV + fL * fV * (rr - 1.f));

        float fss = lerp(1.f, fss90, fL) * lerp(1.f, fss90, fV);
        float ss = 1.25f * (fss * (1.f / (wi.z + wo.z) - .5f) + .5f);

        // NOTE: InvPi * (1.f - specTrans) * (1.f - diffTrans) cancels
        float3 diffuse = cdLin * lerp(fd + retroReflection, ss, flatness) + fSheen;
        value = diffuse;
    }
    else if (u.x < probDiffReflect + probDiffRefract)
    {
        u.x = (u.x - probDiffReflect) / probDiffRefract;
        dir = -SampleCosineHemisphere(u);
        value = cdLin;
    }
    throughput *= cdLin;
    return dir;
}

#endif

#ifndef DISNEY_BSDF_HLSLI
#define DISNEY_BSDF_HLSLI

#include "../sampling.hlsli"
#include "../../rt/shader_interop/hit_shaderinterop.h"

float SchlickFresnel(float u)
{
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m;
}

float3 SampleGGXVNDF(float3 w, float2 u, float alphaX, float alphaY)
{
    float3 wh = normalize(float3(alphaX * w.x, alphaY * w.y, w.z));
    wh         = wh.z < 0 ? -wh : wh;
    float2 p  = SampleUniformDiskPolar(u);
    float3 T1 =
        wh.z < 0.99999f ? normalize(cross(float3(0, 0, 1), wh)) :  float3(1, 0, 0);
    float3 T2 = cross(wh, T1);
    float h = sqrt(1 - p.x * p.x);
    p.y        = lerp((1.f + wh.z) / 2.f, h, p.y);

    // Project point to hemisphere, transform to ellipsoid.
    float pz = sqrt(max(0.f, 1 - p.x * p.x - p.y * p.y));
    float3 nh  = p.x * T1 + p.y * T2 + pz * wh;
    return normalize(float3(alphaX * nh.x, alphaY * nh.y, max(1e-6f, nh.z)));
}

float GTR1(float cosH, float a)
{
    if (a >= 1) return InvPi;
    float a2 = a * a;
    float result = (a2 - 1) / (PI * log2(a2) * (1.f + (a2 - 1.f) * Sqr(cosH)));
    return result;
}

float GTR2Aniso(float3 wm, float alphaX, float alphaY)
{
    return 1.f /
           (PI * alphaX * alphaY * (Sqr(wm.x / alphaX) + Sqr(wm.y / alphaY) + Sqr(wm.z)));
}

float SmithLambda(float3 w, float ax, float ay)
{
    float a = (Sqr(w.x * ax) + Sqr(w.y * ay)) / Sqr(w.z);
    float lambda = (-1 + sqrt(1 + 1 / a)) / 2.f;
    return lambda;
}

float SmithG1(float3 w, float3 wm, float ax, float ay)
{
    if (dot(w, wm) <= 0.f) return 0.f;

    float lambda = SmithLambda(w, ax, ay);
    float G = 1 / (1 + lambda);
    return G;
}

// TODO: multiply vs height correllated?
float SmithG(float3 wo, float3 wi, float ax, float ay)
{
    return 1 / (1 + SmithLambda(wo, ax, ay) + SmithLambda(wi, ax, ay));
}

float GGXPDF(float3 w, float3 wm, float ax, float ay)
{
    float result = SmithG1(w, wm, ax, ay) / abs(w.z) * GTR2Aniso(wm, ax, ay) * abs(dot(w, wm));
    return result;
}

void CalculateAnisotropicRoughness(float anisotropic, float roughness, out float ax, out float ay)
{
    float aspect = sqrt(1 - .9f * anisotropic);
    ax = roughness * roughness / aspect;
    ay = roughness * roughness * aspect;
}

float3 SampleDisney(GPUMaterial material, float2 u, float3 baseColor, inout float3 throughput, float3 wo) 
{
    // GTR2
    float F = FrDielectric(wo.z, material.ior);
    float probSpecReflect = (1.f - material.metallic) * material.specTrans * F;
    float probSpecRefract = (1.f - material.metallic) * material.specTrans * (1 - F);
    float probClearCoat = 0.25f * material.clearcoat;
    float probDiffuse = (1.f - material.metallic) * (1.f - material.specTrans);

    float totalProb = probSpecReflect + probSpecRefract + probClearCoat + probDiffuse;
    probSpecReflect /= totalProb;
    probSpecRefract /= totalProb;
    probClearCoat /= totalProb;
    probDiffuse /= totalProb;

    float3 wi;
    float3 value;

    float alphaX, alphaY;
    CalculateAnisotropicRoughness(material.anisotropic, material.roughness, alphaX, alphaY);
    float3 wm = SampleGGXVNDF(wo, u, alphaX, alphaY);
    float D = GTR2Aniso(wm, alphaX, alphaY);
        
    float cdLum = .3 * baseColor.x + .6 * baseColor.y + .1 * baseColor.z;
    float3 tint = cdLum > 0 ? baseColor / cdLum : 1.f;

    if (u.x < probSpecReflect)
    {
        u.x /= probSpecReflect;
        wi = Reflect(wo, wm);

        float f0 = Sqr((1 - material.ior) / (1 + material.ior));
        float3 r0 = f0 * lerp(1.f, tint, material.specularTint);
        r0 = lerp(r0, baseColor, material.metallic);
        float fR = SchlickFresnel(dot(wi, wm));
        float3 metallicFresnel = r0 + (1 - r0) * fR;
        float3 disneyFresnel = lerp(F, metallicFresnel, material.metallic);

        float G = SmithG(wi, wo, alphaX, alphaY);
        float pdf = GGXPDF(wo, wm, alphaX, alphaY) * probSpecReflect / (4 * abs(dot(wo, wm)));
        value = D * G * disneyFresnel / (4 * abs(wo.z));
        value *= (1.f - material.metallic) * (1.f - material.specTrans);
        value /= pdf;
    }
    else if (u.x < probSpecReflect + probSpecRefract)
    {
        u.x = (u.x - probSpecReflect) / probSpecRefract;
        float etap;
        bool success = Refract(wo, wm, material.ior, etap, wi);
        if (!success)
        {
            return 0;
        }

        float G = SmithG(wi, wo, alphaX, alphaY);
        float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
        float dwm_dwi = abs(dot(wi, wm)) / denom;
        float pdf = GGXPDF(wo, wm, alphaX, alphaY) * dwm_dwi * probSpecRefract;

        value = sqrt(baseColor) * abs(dot(wo, wm)) * abs(dot(wi, wm)) * D * G * (1 - F) / (abs(wo.z) * denom);
        value /= (pdf * Sqr(etap));
    }
    else if (u.x < probSpecReflect + probSpecRefract + probClearCoat)
    {
        u.x = (u.x - probSpecReflect - probSpecRefract) / probClearCoat;

        float a = 0.25f;
        float a2 = a * a;
        float cosTheta = sqrt(max(0.f, (1.f - pow(a2, 1 - u.x)) / (1 - a2)));
        float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        float phi = 2 * PI * u.y;
        float3 wm = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        wm = dot(wm, wo) < 0.f ? -wm : wm;
        wi = Reflect(wo, wm);
        if (dot(wi, wo) < 0.f) return 0;

        float3 h = normalize(wi + wo);
        // TODO
        D = GTR1(wm.z, lerp(0.1f, 0.001f, material.clearcoatGloss));
        F = lerp(0.04f, 1.f, SchlickFresnel(dot(wi, h)));
        float G = SmithG(wi, wo, .25f, .25f);
        float pdf = D / (4.f * dot(wo, wm));

        value = .25f * material.clearcoat * D * G * F;
        value /= pdf;
    }
    else 
    {
        u.x = (u.x - (1.f - probDiffuse)) / probDiffuse;
        wi = SampleCosineHemisphere(u);
        wi = dot(wi, wo) < 0.f ? -wi : wi;

        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);

        float fH = SchlickFresnel(wiDotH);
        float3 cSheen = lerp(1.f, tint, material.sheenTint);
        float3 fSheen = fH * material.sheen * cSheen;

        float fL = SchlickFresnel(wi.z); 
        float fV = SchlickFresnel(wo.z);

        float fd = (1 - 0.5f * fL) * (1 - 0.5f * fV);
        float rr = 2 * material.roughness * wiDotH * wiDotH;
        float retroReflection = rr * (fL + fV + fL * fV * (rr - 1.f));

        // NOTE: cosine/InvPi from pdf and rendering equation cancels
        float3 diffuse = baseColor * (fd + retroReflection) + fSheen;
        value = diffuse;
        value /= probDiffuse;
    }
    throughput *= value;
    return wi;
}

float3 SampleDisneyThin(float2 u, float3 baseColor, inout float3 throughput, float3 wo) 
{
    float sheen = 0.15f;
    float sheenTint = 0.3f;
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
    // TODO: specular BSDF
    if (u.x < probDiffReflect)
    {
        u.x = u.x / probDiffReflect;
        dir = SampleCosineHemisphere(u);

        float3 wi = dir;
        float3 h = normalize(wi + wo);
        float wiDotH = dot(wi, h);

        float fH = SchlickFresnel(wiDotH);
        float cdLum = .3 * baseColor.x + .6 * baseColor.y + .1 * baseColor.z;
        float3 cTint = cdLum > 0 ? baseColor / cdLum : 1.f;
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

        float3 diffuse = baseColor * lerp(fd + retroReflection, ss, flatness) + fSheen;
        value = (1.f - specTrans) * (1.f - diffTrans) * diffuse;
        value /= probDiffReflect;
    }
    else if (u.x < probDiffReflect + probDiffRefract)
    {
        u.x = (u.x - probDiffReflect) / probDiffRefract;
        dir = -SampleCosineHemisphere(u);
        value = baseColor;
        value *= (1.f - specTrans) * diffTrans;
        value /= probDiffRefract;
    }
    throughput *= value;
    return dir;
}

#endif

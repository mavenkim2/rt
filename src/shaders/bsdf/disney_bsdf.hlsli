#ifndef DISNEY_BSDF_HLSLI
#define DISNEY_BSDF_HLSLI

#include "../sampling.hlsli"
#include "bxdf.hlsli"
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

float3 SampleDisney(GPUMaterial material, float3 u, float3 baseColor, inout float3 throughput, float3 wo, out float outPdf) 
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
    float3 value = 1.f;

    float alphaX, alphaY;
    CalculateAnisotropicRoughness(material.anisotropic, material.roughness, alphaX, alphaY);
    float3 wm = material.roughness == 0.f ? float3(0, 0, 1) : SampleGGXVNDF(wo, u.xy, alphaX, alphaY);
    float D = GTR2Aniso(wm, alphaX, alphaY);
        
    float cdLum = .3 * baseColor.x + .6 * baseColor.y + .1 * baseColor.z;
    float3 tint = cdLum > 0 ? baseColor / cdLum : 1.f;

    if (u.z < probSpecReflect)
    {
        wi = Reflect(wo, wm);

        if (material.roughness != 0.f)
        {
            float f0 = Sqr((1 - material.ior) / (1 + material.ior));
            float3 r0 = f0 * lerp(1.f, tint, material.specularTint);
            r0 = lerp(r0, baseColor, material.metallic);
            float fR = SchlickFresnel(dot(wi, wm));
            float3 metallicFresnel = r0 + (1 - r0) * fR;
            float3 disneyFresnel = lerp(F, metallicFresnel, material.metallic);
            float G = SmithG(wi, wo, alphaX, alphaY);
            float pdf = GGXPDF(wo, wm, alphaX, alphaY) * probSpecReflect / (4 * abs(dot(wo, wm)));
            value = D * G * disneyFresnel / (4 * abs(wo.z));
            value *= (1.f - material.metallic) * material.specTrans;
            value /= pdf;
            outPdf = pdf;
        }
    }
    else if (u.z < probSpecReflect + probSpecRefract)
    {
        float etap;
        bool success = Refract(wo, wm, 1.1, etap, wi);
        if (!success)
        {
            return 0;
        }

        if (material.roughness != 0.f)
        {
            float G = SmithG(wi, wo, alphaX, alphaY);
            float denom = Sqr(dot(wi, wm) + dot(wo, wm) / etap);
            float dwm_dwi = abs(dot(wi, wm)) / denom;
            float pdf = GGXPDF(wo, wm, alphaX, alphaY) * dwm_dwi * probSpecRefract;
            value = sqrt(baseColor) * abs(dot(wo, wm)) * abs(dot(wi, wm)) * D * G * (1 - F) / (abs(wo.z) * denom);
            value *= (1.f - material.metallic) * material.specTrans;
            value /= (pdf * Sqr(etap));
            outPdf = pdf;
        }
        else 
        {
            value /= Sqr(etap);
        }
    }
    else if (u.z < probSpecReflect + probSpecRefract + probClearCoat)
    {
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
        float pdf = D / (4.f * dot(wo, wm)) * probClearCoat;

        value = .25f * material.clearcoat * D * G * F;
        value /= pdf;
        outPdf = pdf;
    }
    else 
    {
        u.z = (u.z - (probSpecReflect + probSpecRefract + probClearCoat)) / probDiffuse;

        float diffTrans = material.diffTrans / 2.f;
        float probTransmit = material.thin ? diffTrans : 0.f;
        float probReflect = material.thin ? 1 - diffTrans : 1.f;

        if (material.thin && u.z < probTransmit)
        {
            wi = -SampleCosineHemisphere(u.xy);
            value = baseColor;
            value *= (1.f - material.specTrans) * diffTrans * (1.f - material.metallic);
            value /= (probDiffuse * probTransmit);
            outPdf = (probDiffuse * probTransmit * InvPi * abs(wi.z));
        }
        else 
        {
            wi = SampleCosineHemisphere(u.xy);
            wi = dot(wi, wo) < 0.f ? -wi : wi;

            float3 h = normalize(wi + wo);
            float wiDotH = dot(wi, h);

            float fH = SchlickFresnel(wiDotH);
            float3 cSheen = lerp(1.f, tint, material.sheenTint);
            float3 fSheen = fH * material.sheen * cSheen;

            float fss90 = wiDotH * wiDotH * material.roughness;
            float fL = SchlickFresnel(wi.z); 
            float fV = SchlickFresnel(wo.z);

            float fd = (1 - 0.5f * fL) * (1 - 0.5f * fV);
            float rr = 2 * material.roughness * wiDotH * wiDotH;
            float retroReflection = rr * (fL + fV + fL * fV * (rr - 1.f));

            float fss = lerp(1.f, fss90, fL) * lerp(1.f, fss90, fV);
            float ss = 1.25f * (fss * (1.f / (wi.z + wo.z) - .5f) + .5f);

            float flatness = material.thin ? 0.f : material.flatness;
            float val = lerp(fd + retroReflection, ss, flatness);

            // NOTE: cosine/InvPi from pdf and rendering equation cancels
            float3 diffuse = baseColor * val + fSheen;
            value = diffuse * (1.f - material.specTrans) * (1.f - diffTrans) * (1.f - material.metallic);
            value /= probDiffuse * probReflect;
            outPdf = (probDiffuse * probReflect * InvPi * abs(wi.z));
        }
    }
    throughput *= value;
    return normalize(wi);
}

float3 EvaluateDisney(GPUMaterial material, float3 baseColor, float3 wo, float3 wi, out float outPdf) 
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

    float3 value = 0.f;
    float totalPdf = 0.f;

    float alphaX, alphaY;
    CalculateAnisotropicRoughness(material.anisotropic, material.roughness, alphaX, alphaY);

    float cosThetaO = wo.z;
    float cosThetaI = wi.z;
    bool reflect = wo.z * wi.z > 0.f;
    float etaP = reflect ? 1.f : (cosThetaO > 0.f ? material.ior : (1.f / material.ior));
    float3 wm = normalize(wi * etaP + wo);
    wm = wm.z < 0.f ? -wm : wm;
    float D = GTR2Aniso(wm, alphaX, alphaY);
        
    float cdLum = .3 * baseColor.x + .6 * baseColor.y + .1 * baseColor.z;
    float3 tint = cdLum > 0 ? baseColor / cdLum : 1.f;

    {
        float3 h = normalize(wi + wo);
        D = GTR1(wm.z, lerp(0.1f, 0.001f, material.clearcoatGloss));
        F = lerp(0.04f, 1.f, SchlickFresnel(dot(wi, h)));
        float G = SmithG(wi, wo, .25f, .25f);
        float pdf = probClearCoat * D / (4.f * dot(wo, wm));
        totalPdf += pdf;
        value += .25f * material.clearcoat * D * G * F;
    }
    {
        float diffTrans = material.thin ? material.diffTrans / 2.f : 0.f;
        float probTransmit = diffTrans;
        float probReflect = 1.f - diffTrans;

        if (!reflect)
        {
            value = baseColor;
            value *= (1.f - material.specTrans) * diffTrans * (1.f - material.metallic);
            value /= (probDiffuse * probTransmit);
            outPdf = (probDiffuse * probTransmit * InvPi * abs(wi.z));
        }
        else 
        {
            float3 h = normalize(wi + wo);
            float wiDotH = dot(wi, h);

            float fH = SchlickFresnel(wiDotH);
            float3 cSheen = lerp(1.f, tint, material.sheenTint);
            float3 fSheen = fH * material.sheen * cSheen;

            float fss90 = wiDotH * wiDotH * material.roughness;
            float fL = SchlickFresnel(wi.z); 
            float fV = SchlickFresnel(wo.z);

            float fd = (1 - 0.5f * fL) * (1 - 0.5f * fV);
            float rr = 2 * material.roughness * wiDotH * wiDotH;
            float retroReflection = rr * (fL + fV + fL * fV * (rr - 1.f));

            float fss = lerp(1.f, fss90, fL) * lerp(1.f, fss90, fV);
            float ss = 1.25f * (fss * (1.f / (wi.z + wo.z) - .5f) + .5f);

            float flatness = material.thin ? 0.f : material.flatness;
            float val = lerp(fd + retroReflection, ss, flatness);

            // NOTE: cosine/InvPi from pdf and rendering equation cancels
            float3 diffuse = InvPi * baseColor * val + fSheen;
            value += (1.f - material.metallic) * (1.f - diffTrans) * (1.f - material.specTrans) * diffuse;
            totalPdf += (probDiffuse * probReflect * InvPi * abs(wi.z));
        }
    }
    outPdf = totalPdf;

    return value;
}

#endif

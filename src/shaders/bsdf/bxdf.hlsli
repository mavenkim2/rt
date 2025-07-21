#ifndef BXDF_HLSLI_
#define BXDF_HLSLI_

float3 Reflect(float3 w, float3 n)
{
    return 2 * dot(n, w) * n - w;
}

bool Refract(float3 wi, float3 n, float eta, out float etap, out float3 wt)
{
    float cosTheta_i = dot(wi, n);

    bool mask = cosTheta_i > 0;
    if (!mask)
    {
        n            = -n;
        eta          = 1 / eta;
        cosTheta_i   = -cosTheta_i;
    }

    float sin2Theta_i = max(0.f, 1 - cosTheta_i * cosTheta_i);
    float sin2Theta_t = sin2Theta_i / (eta * eta);

    if (sin2Theta_t >= 1)
    { 
        etap = 0;
        wt = float3(0, 0, 0);
        return false;
    }

    float cosTheta_t = sqrt(max(1 - sin2Theta_t, 0));
    etap = eta;
    wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
    return true;
}

float FrDielectric(float cosTheta_i, float eta)
{
    cosTheta_i           = clamp(cosTheta_i, -1.f, 1.f);
    if (cosTheta_i < 0)
    {
        cosTheta_i           = -cosTheta_i;
        eta                  = 1 / eta;
    }
    float sin2Theta_i = max(0.f, 1 - cosTheta_i * cosTheta_i);
    float sin2Theta_t = sin2Theta_i / (eta * eta);
    float cosTheta_t  = sqrt(max(1 - sin2Theta_t, 0));
    float rParallel   = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    float rPerp       = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    float Fr          = 0.5f * (rPerp * rPerp + rParallel * rParallel);
    return Fr;
}

#endif

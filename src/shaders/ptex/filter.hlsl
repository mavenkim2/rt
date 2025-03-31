#include "../common.hlsli"

struct PtexBicubicFilter
{
    float _coeffs[7];

    float3 Eval(int firstChan, int nChannels,
                int faceid, float u, float v,
                float uw1, float vw1, float uw2, float vw2,
                float width, float blur);
    void SplitAndApply(PtexSeparableKernel& k, int faceID, const Ptex::FaceInfo& f);
};

struct PtexSeparableKernel 
{
    Res res;                    // resolution that kernel was built for
    int u, v;                   // uv offset within face data
    int uw, vw;                 // kernel width
    static const int kmax = 10; // max kernel width
    float ku[kmax];
    float kv[kmax];
    int rot;
};

PtexBicubicFilter GetBicubicFilter()
{
    // compute Cubic filter coefficients:
    // abs(x) < 1:
    //   1/6 * ((12 - 9*B - 6*C)*x^3 + (-18 + 12*B + 6*C)*x^2 + (6 - 2*B))
    //   == c[0]*x^3 + c[1]*x^2 + c[2]
    // abs(x) < 2:
    //   1/6 * ((-B - 6*C)*x^3 + (6*B + 30*C)*x^2 + (-12*B - 48*C)*x + (8*B + 24*C))
    //   == c[3]*x^3 + c[4]*x^2 + c[5]*x + c[6]
    // else: 0

    float sharpness = 0;
    float B = 1.0f; - sharpness; // choose C = (1-B)/2
    _coeffs[0] = 1.5f - B;
    _coeffs[1] = 1.5f * B - 2.5f;
    _coeffs[2] = 1.0f - float(1.0/3.0) * B;
    _coeffs[3] = float(1.0/3.0) * B - 0.5f;
    _coeffs[4] = 2.5f - 1.5f * B;
    _coeffs[5] = 2.0f * B - 4.0f;
    _coeffs[6] = 2.0f - float(2.0/3.0) * B;

    //.5 
    //-1 
    //2/3 
    //-1/6
    //1 
    //-2 
    //4/3
}

static float BicubicKernel(float x, float c[7])
{
    x = abs(x);
    if (x < 1.0f)      return (c[0]*x + c[1])*x*x + c[2];
    else if (x < 2.0f) return ((c[3]*x + c[4])*x + c[5])*x + c[6];
    else               return 0.0f;
}

float3 PtexSeparableFilter::Eval(int firstChan, int nChannels,
                                 int faceid, float u, float v,
                                 float uw1, float vw1, float uw2, float vw2,
                                 float width, float blur)
{
    numChannels = 3;
    // Get face info
    const FaceInfo& f = _tx->getFaceInfo(faceid);

    // If neighborhood is constant, just return constant value of face
    if (f.isNeighborhoodConstant()) {
        PtexPtr<PtexFaceData> data ( _tx->getData(faceid, 0) );
        if (data) {
            char* d = (char*) data->getData() + _firstChanOffset;
            Ptex::ConvertToFloat(result, d, _dt, _nchan);
        }
        return;
    }

    // Get filter width
    float uw = abs(uw1) + abs(uw2);
    float vw = abs(vw1) + abs(vw2);

    u = clamp(u, 0, 1);
    v = clamp(v, 0, 1);

    // Build kernel
    PtexSeparableKernel k;
    BuildKernel(k, u, v, uw, vw, f.res);
    k.stripZeros();

    // Check kernel (debug only)
    //assert(k.uw > 0 && k.vw > 0);
    //assert(k.uw <= PtexSeparableKernel::kmax && k.vw <= PtexSeparableKernel::kmax);

    float weight = k.weight();

    float3 result = 0;
    // Apply to faces
    splitAndApply(k, faceid, f);

    float scale = 1.0f / weight;
    result *= scale;
}

void BuildKernel()
{
    BuildKernelAxis();
    BuildKernelAxis();
}

void BuildKernelAxis(int8_t& k_ureslog2, int& k_u, int& k_uw, float* ku,
                     float u, float uw, int f_ureslog2)
{
    // build 1 axis (note: "u" labels may repesent either u or v axis)

    // clamp filter width to no smaller than a texel
    uw = max(uw, ReciprocalPow2(f_ureslog2));

    // compute desired texture res based on filter width
    k_ureslog2 = (int8_t)PtexUtils::calcResFromWidth(uw);
    int resu = 1 << k_ureslog2;
    float uwlo = 1.0f/(float)resu; // smallest filter width for this res

    // compute lerp weights (amount to blend towards next-lower res)
    float lerp2 = _options.lerp ? (uw-uwlo)/uwlo : 0;
    float lerp1 = 1.0f-lerp2;

    // adjust for large filter widths
    if (uw >= .25f) {
        if (uw < .5f) {
            k_ureslog2 = 2;
            float upix = u * 4.0f - 0.5f;
            int u1 = int(ceil(upix - 2));
            int u2 = int(ceil(upix + 2));
            u1 = u1 & ~1;       // round down to even pair
            u2 = (u2 + 1) & ~1; // round up to even pair
            k_u = u1;
            k_uw = u2-u1;
            float x1 = (float)u1-upix;
            for (int i = 0; i < k_uw; i+=2) {
                float xa = x1 + (float)i, xb = xa + 1.0f, xc = (xa+xb)*0.25f;
                // spread the filter gradually to approach the next-lower-res width
                // at uw = .5, s = 1.0; at uw = 1, s = 0.8
                float s = 1.0f/(uw + .75f);
                float ka = BicubicKernel(xa, _c), kb = BicubicKernel(xb, _c), kc = blur(xc*s);
                ku[i] = ka * lerp1 + kc * lerp2;
                ku[i+1] = kb * lerp1 + kc * lerp2;
            }
            return;
        }
        else if (uw < 1) {
            k_ureslog2 = 1;
            float upix = u * 2.0f - 0.5f;
            k_u = int(PtexUtils::floor(u - .5f))*2;
            k_uw = 4;
            float x1 = (float)k_u-upix;
            for (int i = 0; i < k_uw; i+=2) {
                float xa = x1 + (float)i;
                float xb = xa + 1.0f;
                float xc = (xa + xb) * 0.5f;
                float s = 1.0f / (uw * 1.5f + .5f);
                float ka = Blur(xa * s);
                float kb = Blur(xb * s);
                float kc = Blur(xc * s);
                ku[i] = ka * lerp1;// + kc * lerp2;
                ku[i+1] = kb * lerp1;// + kc * lerp2;
            }
            return;
        }
        else {
            k_ureslog2 = 0;
            float upix = u - .5f;
            k_uw = 2;
            float ui = floor(upix);
            k_u = int(ui);
            ku[0] = Blur(upix-ui);
            ku[1] = 1-ku[0];
            return;
        }
    }

    // convert from normalized coords to pixel coords
    float upix = u * (float)resu - 0.5f;
    float uwpix = uw * (float)resu;

    // find integer pixel extent: [u,v] +/- [2*uw,2*vw]
    // (kernel width is 4 times filter width)
    float dupix = 2.0f*uwpix;
    int u1 = int(ceil(upix - dupix)), u2 = int(ceil(upix + dupix));

    k_u = u1;
    k_uw = u2-u1;
    // compute kernel weights
    float x1 = ((float)u1-upix)/uwpix, step = 1.0f/uwpix;
    for (int i = 0; i < k_uw; i++) ku[i] = BicubicKernel(x1 + (float)i*step, _c);
}

float Blur(float x)
{
    x = abs(x);
    return x < 1.0f ? (2.0f * x - 3.0f) * x * x + 1.0f : 0.0f;
}

void PtexBicubicFilter::SplitAndApply(PtexSeparableKernel& k, int faceid, const Ptex::FaceInfo& f)
{
    // do we need to split? (i.e. does kernel span an edge?)
    bool splitR = (k.u+k.uw > k.res.u()), splitL = (k.u < 0);
    bool splitT = (k.v+k.vw > k.res.v()), splitB = (k.v < 0);

    if (splitR || splitL || splitT || splitB) {
        PtexSeparableKernel ka, kc;
        if (splitR) {
            if (f.adjface(e_right) >= 0) {
                k.splitR(ka);
                if (splitT) {
                    if (f.adjface(e_top) >= 0) {
                        ka.splitT(kc);
                        applyToCorner(kc, faceid, f, e_top);
                    }
                    else ka.mergeT(_vMode);
                }
                if (splitB) {
                    if (f.adjface(e_bottom) >= 0) {
                        ka.splitB(kc);
                        applyToCorner(kc, faceid, f, e_right);
                    }
                    else ka.mergeB(_vMode);
                }
                applyAcrossEdge(ka, faceid, f, e_right);
            }
            else k.mergeR(_uMode);
        }
        if (splitL) {
            if (f.adjface(e_left) >= 0) {
                k.splitL(ka);
                if (splitT) {
                    if (f.adjface(e_top) >= 0) {
                        ka.splitT(kc);
                        applyToCorner(kc, faceid, f, e_left);
                    }
                    else ka.mergeT(_vMode);
                }
                if (splitB) {
                    if (f.adjface(e_bottom) >= 0) {
                        ka.splitB(kc);
                        applyToCorner(kc, faceid, f, e_bottom);
                    }
                    else ka.mergeB(_vMode);
                }
                applyAcrossEdge(ka, faceid, f, e_left);
            }
            else k.mergeL(_uMode);
        }
        if (splitT) {
            if (f.adjface(e_top) >= 0) {
                k.splitT(ka);
                applyAcrossEdge(ka, faceid, f, e_top);
            }
            else k.mergeT(_vMode);
        }
        if (splitB) {
            if (f.adjface(e_bottom) >= 0) {
                k.splitB(ka);
                applyAcrossEdge(ka, faceid, f, e_bottom);
            }
            else k.mergeB(_vMode);
        }
    }

    // do local face
    apply(k, faceid, f);
}





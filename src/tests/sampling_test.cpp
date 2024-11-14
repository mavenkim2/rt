namespace rt
{
#if 0
void SphericalSampleTest()
{
    int count  = 1024 * 1024;
    Vec3f v[4] = {Vec3f(4, 1, 1), Vec3f(6, 1, -2), Vec3f(4, 4, 1),
                  Vec3f(6, 4, -2)};
    f32 A      = Length(v[0] - v[1]) * Length(v[0] - v[2]);
    Vec3f N    = Normalize(Cross(v[1] - v[0], v[2] - v[0]));
    Vec3f p(.5, -.4, .7);

    // Integrate this function over the projection of the quad given by
    // |v| at the unit sphere surrounding |p|.
    auto f = [](Vec3f p) { return p.x * p.y * p.z; };

    LaneIF32 sphSum = 0, areaSum = 0;
    ZSobolSampler sampler(1024, Vec2i(0), RandomizeStrategy::FastOwen);
    sampler.StartPixelSample(Vec2i(0), 0);
    for (int i = 0; i < count; i++)
    {
        Vec2f u = sampler.Get2D();
        LaneIF32 pdf;
        Vec3f pq = SampleSphericalRectangle(p, v[0], v[1] - v[0], v[2] - v[0], u, &pdf);
        sphSum += f(pq) / pdf;

        pq        = Lerp(u[1], Lerp(u[0], v[0], v[1]), Lerp(u[0], v[2], v[3]));
        pdf       = 1.f / A;
        Vec3f pqp = p - pq; // pq - p;
        areaSum += f(pq) * AbsDot(N, Normalize(pqp)) / (pdf * LengthSquared(pqp));
    }
    LaneIF32 sphInt  = sphSum / f32(count);
    LaneIF32 areaInt = areaSum / f32(count);

    if (Abs(areaInt - sphInt) < 1e-3)
    {
        printf("yay");
    }
    else
    {
        printf("nay");
    }
    printf("%f, %f\n", sphInt.value, areaInt.value);
}
#endif
} // namespace rt

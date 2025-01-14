struct CovarianceMatrix
{
    f32 values[10];

    // * C =  ( 0  1  3  6)
    // *      ( *  2  4  7)
    // *      ( *  *  5  8)
    // *      ( *  *  *  9)

    void Travel(f32 t) { values[3] = }
};

void ConvertCovarianceToRayDifferentials(Vec3f &dpdx, Vec3f &dpdy)
{
    // Compute eigenvectors
}

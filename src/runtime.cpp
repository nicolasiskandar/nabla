#include "export.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" DLLEXPORT double *nabla_solve(const double *A, const double *b,
                                         int64_t n)
{
    auto *LU = (double *)malloc(n * n * sizeof(double));
    memcpy(LU, A, n * n * sizeof(double));
    auto *pivot = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        pivot[i] = i;

    for (int k = 0; k < n - 1; k++)
    {
        int best = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(LU[i * n + k]) > fabs(LU[best * n + k]))
                best = i;
        if (best != k)
        {
            for (int j = 0; j < n; j++)
            {
                double t = LU[k * n + j];
                LU[k * n + j] = LU[best * n + j];
                LU[best * n + j] = t;
            }
            int tp = pivot[k];
            pivot[k] = pivot[best];
            pivot[best] = tp;
        }
        for (int i = k + 1; i < n; i++)
        {
            LU[i * n + k] /= LU[k * n + k];
            for (int j = k + 1; j < n; j++)
                LU[i * n + j] -= LU[i * n + k] * LU[k * n + j];
        }
    }

    auto *x = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        x[i] = b[pivot[i]];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            x[i] -= LU[i * n + j] * x[j];

    for (int i = n - 1; i >= 0; i--)
    {
        for (int j = i + 1; j < n; j++)
            x[i] -= LU[i * n + j] * x[j];
        x[i] /= LU[i * n + i];
    }

    free(LU);
    free(pivot);
    return x;
}

extern "C" DLLEXPORT double *nabla_inv(const double *A, int64_t n)
{
    auto *inv = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        auto *e = (double *)calloc(n, sizeof(double));
        e[i] = 1.0;
        double *col = nabla_solve(A, e, n);
        for (int j = 0; j < n; j++)
            inv[j * n + i] = col[j];
        free(e);
        free(col);
    }
    return inv;
}

extern "C" DLLEXPORT double nabla_det(const double *A, int64_t n)
{
    auto *LU = (double *)malloc(n * n * sizeof(double));
    memcpy(LU, A, n * n * sizeof(double));
    double det = 1.0;

    for (int k = 0; k < n - 1; k++)
    {
        int best = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(LU[i * n + k]) > fabs(LU[best * n + k]))
                best = i;
        if (best != k)
        {
            for (int j = 0; j < n; j++)
            {
                double t = LU[k * n + j];
                LU[k * n + j] = LU[best * n + j];
                LU[best * n + j] = t;
            }
            det = -det;
        }
        for (int i = k + 1; i < n; i++)
        {
            double factor = LU[i * n + k] / LU[k * n + k];
            for (int j = k + 1; j < n; j++)
                LU[i * n + j] -= factor * LU[k * n + j];
        }
    }

    for (int i = 0; i < n; i++)
        det *= LU[i * n + i];

    free(LU);
    return det;
}

static int jacobi_sweep(double *A, double *eig, int64_t n)
{
    int converged = 1;
    for (int p = 0; p < n - 1; p++)
        for (int q = p + 1; q < n; q++)
        {
            double apq = fabs(A[p * n + q]);
            if (apq < 1e-15)
                continue;
            converged = 0;
            double theta = (A[q * n + q] - A[p * n + p]) / (2.0 * A[p * n + q]);
            double t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
            if (theta < 0)
                t = -t;
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;
            double tau = s / (1.0 + c);
            double app = A[p * n + p], aqq = A[q * n + q];
            double apq_val = A[p * n + q];
            A[p * n + p] = app - t * apq_val;
            A[q * n + q] = aqq + t * apq_val;
            A[p * n + q] = 0.0;
            A[q * n + p] = 0.0;
            for (int r = 0; r < p; r++)
            {
                double arp = A[r * n + p], arq = A[r * n + q];
                A[r * n + p] = arp - s * (arq + tau * arp);
                A[r * n + q] = arq + s * (arp - tau * arq);
            }
            for (int r = p + 1; r < q; r++)
            {
                double apr = A[p * n + r], arq = A[r * n + q];
                A[r * n + p] = apr - s * (arq + tau * apr);
                A[r * n + q] = arq + s * (apr - tau * arq);
            }
            for (int r = q + 1; r < n; r++)
            {
                double apr = A[p * n + r], aqr = A[q * n + r];
                A[p * n + r] = apr - s * (aqr + tau * apr);
                A[q * n + r] = aqr + s * (apr - tau * aqr);
            }
        }
    return converged;
}

extern "C" DLLEXPORT double *nabla_eig(const double *A, int64_t n)
{
    auto *work = (double *)malloc(n * n * sizeof(double));
    memcpy(work, A, n * n * sizeof(double));
    for (int iter = 0; iter < 100; iter++)
        if (jacobi_sweep(work, nullptr, n))
            break;
    auto *eig = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        eig[i] = work[i * n + i];
    free(work);
    return eig;
}

#pragma once
#include "ggl.h"


class Interpolator
{
public:
    struct TRIANGLE
    {
        vec3 v[3];
        TRIANGLE() = default;
        TRIANGLE(vec3 v1, vec3 v2, vec3 v3)
        {
            v[0] = v1;
            v[1] = v2;
            v[2] = v3;
        }
    };
    static double calcTriangleArea(TRIANGLE& tri)
    {
        auto A = tri.v[0];
        auto B = tri.v[1];
        auto C = tri.v[2];

        auto AB = B - A;
        auto AC = C - A;

        auto tmp_cross = YOSEF_CROSS(AB, AC);
        auto tmp_length = YOSEF_LENGTH(tmp_cross);
        return 0.5 * tmp_length;
    }
    /*
    static float calcSphereDist(const vec2f& v1, const vec2f& v2)
    {
        auto theta1 = v1.x;
        auto phi1 = v1.y;
        auto theta2 = v2.x;
        auto phi2 = v2.y;

        auto r = EARTH_RADIUS * 1000.0f;
        float d = r * acos(sin(theta1) * sin(theta2) + cos(theta1) * cos(theta2) + cos(abs(phi2 - phi1)));
        return d;
    }

    static float calcLinearInterpolate(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    static vec2f calcLinearInterpolate(const vec2f& a, const vec2f& b, float t)
    {
        return vec2f(
            Interpolator::calcLinearInterpolate(a.x, b.x, t),
            Interpolator::calcLinearInterpolate(a.y, b.y, t)
        );
    }
    
    static vec3f calcLinearInterpolate(const vec3f& a, const vec3f& b, float t)
    {
        return vec3f(
            Interpolator::calcLinearInterpolate(a.x, b.x, t),
            Interpolator::calcLinearInterpolate(a.y, b.y, t),
            Interpolator::calcLinearInterpolate(a.z, b.z, t)
        );
    }
    static float calcBiLinearInterpolate(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y) {
        float x2x1 = x2 - x1, y2y1 = y2 - y1, x2x = x2 - x, y2y = y2 - y, yy1 = y - y1, xx1 = x - x1;
        return 1.0f / (x2x1 * y2y1) * (
            q11 * x2x * y2y +
            q21 * xx1 * y2y +
            q12 * x2x * yy1 +
            q22 * xx1 * yy1
            );
    }*/

    static void calcTriangleBarycentric(const vec3& p, TRIANGLE* tri, double& u, double& v, double& w)
    {
        auto v0 = tri->v[1] - tri->v[0];
        auto v1 = tri->v[2] - tri->v[0];
        auto v2 = p - tri->v[0];
        double d00 = YOSEF_DOT(v0, v0);
        double d01 = YOSEF_DOT(v0, v1);
        double d11 = YOSEF_DOT(v1, v1);
        double d20 = YOSEF_DOT(v2, v0);
        double d21 = YOSEF_DOT(v2, v1);
        double denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;
    }

    static double triangle_area(const vec3& a, const vec3& b, const vec3& c)
    {
        // Calculate the vectors for two edges of the triangle
        vec3 edge1 = {b.x() - a.x(), b.y() - a.y(), b.z() - a.z()};
        vec3 edge2 = {c.x() - a.x(), c.y() - a.y(), c.z() - a.z()};

        // Calculate the cross product of the two edge vectors
        vec3 cross_product = {
            edge1.y() * edge2.z() - edge1.z() * edge2.y(),
            edge1.z() * edge2.x() - edge1.x() * edge2.z(),
            edge1.x() * edge2.y() - edge1.y() * edge2.x()
        };

        // The area of the triangle is half the magnitude of the cross product vector
        return sqrt(cross_product.x() * cross_product.x() + cross_product.y() * cross_product.y() + cross_product.z() * cross_product.z()) / 2.0f;
    }

    static void CalcPolygonWachspress(const vec3& p, std::vector<vec3>& poly, std::vector<double>& weights)
    {
        int N = poly.size();
        weights.clear();
        weights.resize(N, 0.0);
        double sumweights = 0.0;
        double A_i, A_iplus1, B;
        
        A_iplus1 = triangle_area(poly[N - 1], poly[0], p);
        for(int i = 0; i < N; i++) {
            A_i = A_iplus1;
            A_iplus1 = triangle_area(poly[i], poly[(i + 1) % N], p);
            
            B = triangle_area(poly[(i - 1 + N) % N], poly[i], poly[(i + 1) % N]);
            
            weights[i] = B / (A_i * A_iplus1);
            sumweights += weights[i];
        }
        
        // Normalize the weights
        double recp = 1.0 / sumweights;
        for(int i = 0; i < N; i++) {
            weights[i] *= recp;
        }
    }
    static void CalcPolygonWachspress(const vec3& p, vec3* poly, double* weights, const int vertex_number)
    {
        int N = vertex_number;
        
        for (int i = 0; i < N; i++) {
            weights[i] = 0.0;
        }

        double sumweights = 0.0;
        double A_i, A_iplus1, B;

  
        A_iplus1 = triangle_area(poly[N - 1], poly[0], p);
        for (int i = 0; i < N; i++) {
            A_i = A_iplus1;
            A_iplus1 = triangle_area(poly[i], poly[(i + 1) % N], p);

            B = triangle_area(poly[(i - 1 + N) % N], poly[i], poly[(i + 1) % N]);

            weights[i] = B / (A_i * A_iplus1);
            sumweights += weights[i];
        }

        // Normalize the weights
        double recp = 1.0 / sumweights;
        for (int i = 0; i < N; i++) {
            weights[i] *= recp;
        }
    }
};
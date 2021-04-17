// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: 22 Jan 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>    // std::max

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
    explicit CubicSpline(const Real h = 1) : _dim(2) {
        setSmoothingLen(h);
    }

    void setSmoothingLen(const Real h) {
        const Real h2 = square(h), h3 = h2 * h;
        _h = h;
        _sr = 2e0 * h;
        _c[0] = 2e0 / (3e0 * h);
        _c[1] = 10e0 / (7e0 * M_PI * h2);
        _c[2] = 1e0 / (M_PI * h3);
        _gc[0] = _c[0] / h;
        _gc[1] = _c[1] / h;
        _gc[2] = _c[2] / h;
    }

    Real smoothingLen() const { return _h; }

    Real supportRadius() const { return _sr; }

    Real f(const Real l) const {
        const Real q = l / _h;
        if (q < 1e0) return _c[_dim - 1] * (1e0 - 1.5 * square(q) + 0.75 * cube(q));
        else if (q < 2e0) return _c[_dim - 1] * (0.25 * cube(2e0 - q));
        return 0;
    }

    Real derivative_f(const Real l) const {
        const Real q = l / _h;
        if (q <= 1e0) return _gc[_dim - 1] * (-3e0 * q + 2.25 * square(q));
        else if (q < 2e0) return -_gc[_dim - 1] * 0.75 * square(2e0 - q);
        return 0;
    }

    Real w(const Vec2f &rij) const { return f(rij.length()); }

    Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }

    Vec2f grad_w(const Vec2f &rij, const Real len) const {
        return derivative_f(len) * rij / len;
    }

private:
    unsigned int _dim;
    Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
    explicit SphSolver(
            const Real nu = 0.08, const Real h = 0.5, const Real density = 1e3,
            const Vec2f g = Vec2f(0, -9.8), const Real eta = 0.01, const Real gamma = 7.0) :
            _kernel(h), _nu(nu), _h(h), _d0(density),
            _g(g), _eta(eta), _gamma(gamma) {
        _dt = 0.0073;
        _m0 = _d0 * _h * _h;
        _c = std::fabs(_g.y) / _eta;
        _p0 = 6000000;
    }

    // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
    // the size of f_width, f_height; each cell is sampled with 2x2 particles.
    void initScene(
            const int res_x, const int res_y, const int f_width, const int f_height) {
        _pos.clear();

        _resX = res_x;
        _resY = res_y;

        // set wall for boundary
        _l = 0.5 * _h;
        _r = static_cast<Real>(res_x) - 0.5 * _h;
        _b = 0.5 * _h;
        _t = static_cast<Real>(res_y) - 0.5 * _h;

        // sample a fluid mass
        for (int j = -f_width / 4; j < f_height/4; ++j) {
            for (int i = -f_width / 4; i < f_width/4; ++i) {
                _pos.push_back(Vec2f(res_x / 2 + i + 0.25, 20+j + 0.25));
                _pos.push_back(Vec2f(res_x / 2 + i + 0.75, 20+j + 0.25));
                _pos.push_back(Vec2f(res_x / 2 + i + 0.25, 20+j + 0.75));
                _pos.push_back(Vec2f(res_x / 2 + i + 0.75, 20+j + 0.75));
            }
        }

        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < res_x; ++i) {
                _pos.push_back(Vec2f(i + 0.25,j + 0.25));
                _pos.push_back(Vec2f(i + 0.75,j + 0.25));
                _pos.push_back(Vec2f(i + 0.25,j + 0.75));
                _pos.push_back(Vec2f(i + 0.75,j + 0.75));
            }
        }
        nb_particles = _pos.size();
        /**
        //Create boundary particle
        for (float i = 0; i < res_y - 1; i+=0.5) {
            _pos.push_back(Vec2f(0.5, i + 0.5));
            _pos.push_back(Vec2f(res_x - 0.5, i + 0.5));
        }
        for (float i = 0.5; i < res_x - 1; i+=0.5) {
            _pos.push_back(Vec2f(i + 0.5, res_y - 0.5));
            _pos.push_back(Vec2f(i + 0.5, 0.5));
        }
         **/
        _pos_p = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));

        // make sure for the other particle quantities
        _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _vel_p = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _acc_p = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _p = std::vector<Real>(_pos.size(), 0);
        _d = std::vector<Real>(_pos.size(), 0);
        _d_p = std::vector<Real>(_pos.size(), 0.0f);

        _col = std::vector<float>((_pos.size()) * 4, 1.0); // RGBA
        _vln = std::vector<float>((_pos.size()) * 4, 0.0); // GL_LINES

        updateColor();
    }

    void update() {
        std::cout << '.' << std::flush;
        buildNeighbor();
        computeDensity();
        computePressure();

        _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        applyBodyForce();
        applyPressureForce();
        applyViscousForce();

        updateVelocity();
        updatePosition();

        resolveCollision();

        updateColor();
        if (gShowVel) updateVelLine();
    }

    void update_spisph() {
        int max_iter = 35;
        int iter = 0;
        auto comparison = [](const Vec2f &a, const Vec2f &b) {
            return a.length() < b.length();
        };
        Real _threshold_eta = -0.01 * this->_d0;//1%
        std::cout << '.' << std::flush;
        //We start by building the neighboor
        buildNeighbor();
        //We need some kind of "truth" which is done by computing the density one time at the start
        if (gAppTimer < _dt)
            computeDensity();


        _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        _d_error = std::vector<Real>(_pos.size(), 0.0f);
        //We compute the force, except the pressure one, which we'll just intialize to 0
        applyBodyForce();
        applyViscousForce();
        //Initializing pressure and pressure force to 0
        initializePressure();
        _acc_p = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
        do {
            predictVelocity();
            predictPosition();
            resolvePredictedCollision();
            predictDensity();
            predictDensityVariation();
            Real beta = (this->_dt * this->_dt) * (this->_m0 * this->_m0) * (2 / (_d[0] * _d[0]));
            Real delta = -1 / (beta * deltaDeno());
            for (unsigned int i = 0; i < this->nb_particles; ++i) {
                _p[i] += delta * _d_error[i];
            }
            predictPressureForce();
        } while (averageDensityVariation() < _threshold_eta && iter < max_iter);
        applyPressureForceSPH();
        updateVelocity();
        updatePosition();
        resolveCollision();

        updateColor();
        if (gShowVel) updateVelLine();
    }

    tIndex particleCount() const { return _pos.size(); }

    tIndex boundarCount() const { return _pos_boundary.size(); }

    const Vec2f &position(const tIndex i) const {
        return _pos[i];
    }

    const Vec2f &position_boundary(const tIndex i) const { return _pos_boundary[i]; }

    const float &color(const tIndex i) const { return _col[i]; }

    const float &vline(const tIndex i) const { return _vln[i]; }

    int resX() const { return _resX; }

    int resY() const { return _resY; }

    Real equationOfState(const Real d) {
        return std::max(_p0 * (pow((d / _d0), _gamma) - 1), 0.0);
    }

private:
    void buildNeighbor() {
        int nb_particles = this->particleCount();
        this->_pidxInGrid.clear();
        this->_pidxInGrid.resize(resX() * resY());
        for (int i = 0; i < nb_particles; ++i) {
            Vec2f pos = this->position(i);
            this->_pidxInGrid[this->idx1d(floor(pos.x), floor(pos.y))].push_back(i);
        }
    }

    void computeDensity() {
        for (int i = 0; i < this->particleCount(); i++) {
            Real x_particle = _pos[i].x;
            Real y_particle = _pos[i].y;
            Real radius = _kernel.supportRadius();
            // Compute the neighbours we should take
            int x_lower_bound = floor(-radius + x_particle) >= 0 ? floor(-radius + x_particle) : 0;
            int x_higher_bound =
                    floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
            int y_lower_bound = floor(-radius + y_particle) >= 0 ? floor(-radius + y_particle) : 0;
            int y_higher_bound =
                    floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
            // Start iterating through neighbour
            _d[i] = _eta;
            for (unsigned int x = x_lower_bound; x <= x_higher_bound; x++) {
                for (unsigned int y = y_lower_bound; y <= y_higher_bound; y++) {
                    unsigned int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                    for (unsigned int m = 0; m < nb_particles_in_cell; ++m) {
                        _d[i] += this->_m0 *
                                 this->_kernel.w(this->_pos[i] - this->position(_pidxInGrid[idx1d(x, y)][m]));
                    }
                }
            }
        }
    }

    void predictDensity() {
        for (int i = 0; i < this->particleCount(); i++) {
            Real x_particle = _pos_p[i].x;
            Real y_particle = _pos_p[i].y;
            Real radius = _kernel.supportRadius();
            // Compute the neighbours we should take
            int x_lower_bound = floor(-radius + x_particle) >= 0 ? floor(-radius + x_particle) : 0;
            int x_higher_bound =
                    floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
            int y_lower_bound = floor(-radius + y_particle) >= 0 ? floor(-radius + y_particle) : 0;
            int y_higher_bound =
                    floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
            // Start iterating through neighbour
            _d_p[i] = _eta;
            for (unsigned int x = x_lower_bound; x <= x_higher_bound; x++) {
                for (unsigned int y = y_lower_bound; y <= y_higher_bound; y++) {
                    unsigned int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                    for (unsigned int m = 0; m < nb_particles_in_cell; ++m) {
                        _d_p[i] += this->_m0 *
                                   this->_kernel.w(this->_pos_p[i] - this->_pos_p[_pidxInGrid[idx1d(x, y)][m]]);
                    }
                }
            }
        }
    }

    void predictDensityVariation() {
        for (unsigned int i = 0; i < this->particleCount(); ++i) {
            _d_error[i] = _d_p[i] - _d[i];
        }
    }

    Real deltaDeno() {
        Real x_particle = _pos_p[0].x;
        Real y_particle = _pos_p[0].y;
        Real radius = _kernel.supportRadius();
        Vec2f grad = Vec2f(0.0, 0.0);
        Real gradSquare = 0.0f;
        // Compute the neighbours we should take
        int x_lower_bound = floor(-radius + x_particle) >= 0 ? floor(-radius + x_particle) : 0;
        int x_higher_bound =
                floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
        int y_lower_bound = floor(-radius + y_particle) >= 0 ? floor(-radius + y_particle) : 0;
        int y_higher_bound =
                floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
        // Start iterating through neighbour
        for (unsigned int x = x_lower_bound; x <= x_higher_bound; x++) {
            for (unsigned int y = y_lower_bound; y <= y_higher_bound; y++) {
                unsigned int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                for (unsigned int m = 0; m < nb_particles_in_cell; ++m) {
                    int index = _pidxInGrid[idx1d(x, y)][m];
                    Vec2f xij = this->_pos_p[0] - this->_pos_p[index];
                    if (index == 0 || xij.length() < 0.09)
                        continue;
                    Vec2f gradInter = this->_kernel.grad_w(xij);
                    grad += gradInter;
                    gradSquare += gradInter.dotProduct(gradInter);
                }
            }
        }
        return (-grad.dotProduct(grad) - gradSquare);
    }

    void initializePressure() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            _p[i] = 0.0f;
        }
    }

    void computePressureEstimated() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            _p[i] = equationOfState(_d_p[i]);
        }
    }

    void computePressure() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            _p[i] = equationOfState(_d[i]);
        }
    }


    Real averageDensityVariation() {
        Real avg_var = 0.0f;
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            avg_var += this->_d_error[i];
        }
        return avg_var / this->nb_particles;
    }


    void applyBodyForce() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            _acc[i] += _g;
        }
    }

    void applyPressureForce() {
        Vec2f acc_i = Vec2f(0.0, 0.0);
        for (unsigned int i = 0; i < nb_particles; i++) {
            Real x_particle = _pos[i].x;
            Real y_particle = _pos[i].y;
            Real val1 = _p[i] / (pow(_d[i], 2));
            Real radius = _kernel.supportRadius();
            // Compute the neighbours we should take
            int x_lower_bound = floor(-radius + x_particle) > 0 ? floor(-radius + x_particle) : 0;
            int x_higher_bound =
                    floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
            int y_lower_bound = floor(-radius + y_particle) > 0 ? floor(-radius + y_particle) : 0;
            int y_higher_bound =
                    floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
            // Start iterating through neighbour
            for (int x = x_lower_bound; x <= x_higher_bound; x++) {
                for (int y = y_lower_bound; y <= y_higher_bound; y++) {
                    int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                    for (int m = 0; m < nb_particles_in_cell; ++m) {
                        int index = _pidxInGrid[idx1d(x, y)][m];
                        if (index != i) {
                            Real val2 = _p[index] / (pow(_d[index], 2));
                            Vec2f rij = this->_pos[i] - this->_pos[index];
                            if (rij.length() < 0.01)
                                continue;
                            Vec2f grad = this->_kernel.grad_w(rij);
                            acc_i += this->_m0 * ((val2 + val1) * grad);
                        }
                    }
                }
            }
            _acc[i] -= acc_i;
            acc_i = Vec2f(0, 0);
        }
    }

    void predictPressureForce() {
        Vec2f acc_i = Vec2f(0.0, 0.0);
        for (unsigned int i = 0; i < nb_particles; i++) {
            Real x_particle = _pos[i].x;
            Real y_particle = _pos[i].y;
            Real val1 = _p[i] / (pow(_d[i], 2));
            Real radius = _kernel.supportRadius();
            // Compute the neighbours we should take
            int x_lower_bound = floor(-radius + x_particle) > 0 ? floor(-radius + x_particle) : 0;
            int x_higher_bound =
                    floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
            int y_lower_bound = floor(-radius + y_particle) > 0 ? floor(-radius + y_particle) : 0;
            int y_higher_bound =
                    floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
            // Start iterating through neighbour
            for (int x = x_lower_bound; x <= x_higher_bound; x++) {
                for (int y = y_lower_bound; y <= y_higher_bound; y++) {
                    int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                    for (int m = 0; m < nb_particles_in_cell; ++m) {
                        int index = _pidxInGrid[idx1d(x, y)][m];
                        if (index != i) {
                            Real val2 = _p[index] / (pow(_d[index], 2));
                            Vec2f rij = this->_pos[i] - this->_pos[index];
                            if (rij.length() < 0.01)
                                continue;
                            Vec2f grad = this->_kernel.grad_w(rij);
                            acc_i += this->_m0 * ((val2 + val1) * grad);
                        }
                    }
                }
            }
            _acc_p[i] -= acc_i;
            acc_i = Vec2f(0, 0);
        }
    }

    void applyPressureForceSPH() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            this->_acc[i] += this->_acc_p[i];
        }
    }

    void applyViscousForce() {
        Vec2f acc_i = Vec2f(0.0, 0.0);
        for (unsigned int i = 0; i < nb_particles; i++) {
            Real x_particle = _pos[i].x;
            Real y_particle = _pos[i].y;
            Real radius = _kernel.supportRadius();
            // Compute the neighbours we should take
            int x_lower_bound = floor(-radius + x_particle) > 0 ? floor(-radius + x_particle) : 0;
            int x_higher_bound =
                    floor(radius + x_particle) < this->resX() ? floor(radius + x_particle) : this->resX() - 1;
            int y_lower_bound = floor(-radius + y_particle) > 0 ? floor(-radius + y_particle) : 0;
            int y_higher_bound =
                    floor(radius + y_particle) < this->resY() ? floor(radius + y_particle) : this->resY() - 1;
            // Start iterating through neighbour
            for (int x = x_lower_bound; x <= x_higher_bound; x++) {
                for (int y = y_lower_bound; y <= y_higher_bound; y++) {
                    int nb_particles_in_cell = _pidxInGrid[idx1d(x, y)].size();
                    for (int m = 0; m < nb_particles_in_cell; ++m) {
                        if (_pidxInGrid[idx1d(x, y)][m] != i) {
                            int index = _pidxInGrid[idx1d(x, y)][m];
                            Vec2f uij = this->_vel[i] - this->_vel[index];
                            Vec2f xij = this->_pos[i] - this->_pos[index];
                            if (xij.length() < 0.01)
                                continue;
                            Vec2f grad = this->_kernel.grad_w(xij);
                            Real scalar_prod1 = xij.dotProduct(grad);
                            Real scalar_prod2 = xij.dotProduct(xij) + 0.01 * pow(this->_h, 2);
                            acc_i += (this->_m0 / this->_d[index]) * uij * (scalar_prod1 / scalar_prod2);
                        }
                    }
                }
            }
            _acc[i] += 2 * this->_nu * acc_i;
            acc_i = Vec2f(0, 0);
        }
    }

    void updateVelocity() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            this->_vel[i] += this->_dt * _acc[i];
            this->_vel[i] = this->_vel[i].x > _c || this->_vel[i].y > _c ? Vec2f(_c, _c) : this->_vel[i];
        }
    }

    void predictVelocity() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            this->_vel_p[i] = this->_vel[i] + this->_dt * (_acc[i] + _acc_p[i]);
            this->_vel_p[i] = this->_vel_p[i].x > _c || this->_vel_p[i].y > _c ? Vec2f(_c, _c) : this->_vel_p[i];
        }
    }

    void updatePosition() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            this->_pos[i] += this->_dt * _vel[i];
        }
    }

    void predictPosition() {
        for (unsigned int i = 0; i < this->nb_particles; ++i) {
            this->_pos_p[i] = this->_pos[i] + this->_dt * _vel_p[i];
        }
    }

    // simple collision detection/resolution for each particle
    void resolveCollision() {
        std::vector<tIndex> need_res;
        for (tIndex i = 0; i < nb_particles; ++i) {
            if (_pos[i].x < _l || _pos[i].y < _b || _pos[i].x > _r || _pos[i].y > _t)
                need_res.push_back(i);
        }

        for (std::vector<tIndex>::const_iterator it = need_res.begin(); it < need_res.end();
             ++it) {
            const Vec2f p0 = _pos[*it];
            _pos[*it].x = clamp(_pos[*it].x, _l, _r);
            _pos[*it].y = clamp(_pos[*it].y, _b, _t);
            _vel[*it] = (_pos[*it] - p0) / _dt;
        }
    }

    // simple collision detection/resolution for each particle
    void resolvePredictedCollision() {
        std::vector<tIndex> need_res;
        for (tIndex i = 0; i < nb_particles; ++i) {
            if (_pos_p[i].x < _l || _pos_p[i].y < _b || _pos_p[i].x > _r || _pos_p[i].y > _t)
                need_res.push_back(i);
        }

        for (std::vector<tIndex>::const_iterator it = need_res.begin(); it < need_res.end();
             ++it) {
            const Vec2f p0 = _pos_p[*it];
            _pos_p[*it].x = clamp(_pos[*it].x, _l, _r);
            _pos_p[*it].y = clamp(_pos[*it].y, _b, _t);
            _vel_p[*it] = (_pos[*it] - p0) / _dt;
        }
    }

    void updateColor() {
        for (tIndex i = 0; i < this->nb_particles; ++i) {
            _col[i * 4 + 0] = 0.6;
            _col[i * 4 + 1] = 0.6;
            _col[i * 4 + 2] = _d[i] / _d0;
        }
    }

    void updateVelLine() {
        for (tIndex i = 0; i < particleCount(); ++i) {
            _vln[i * 4 + 0] = _pos[i].x;
            _vln[i * 4 + 1] = _pos[i].y;
            _vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
            _vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
        }
    }

    inline tIndex idx1d(const int i, const int j) { return i + j * resX(); }

    CubicSpline _kernel;

// particle data
    std::vector<Vec2f> _pos;      // position
    std::vector<Vec2f> _vel;      // velocity
    std::vector<Vec2f> _acc;      // acceleration
    std::vector<Real> _p;        // pressure
    std::vector<Real> _d;        // density
    int nb_particles;
// for PCISPH
    std::vector<Real> _d_p;        // density estimated for t+1
    std::vector<Real> _d_error;    // density error estimated for t+1
    std::vector<Vec2f> _vel_p;     // velocity
    std::vector<Vec2f> _pos_p;     // position
    std::vector<Vec2f> _acc_p;     // predict pressure force

    std::vector<Vec2f> _pos_boundary; // boundary particle



    std::vector<std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

    std::vector<float> _col;    // particle color; just for visualization
    std::vector<float> _vln;    // particle velocity lines; just for visualization

// simulation
    Real _dt;                     // time step

    int _resX, _resY;             // background grid resolution

// wall
    Real _l, _r, _b, _t;          // wall (boundary)

// SPH coefficients
    Real _nu;                     // viscosity coefficient, nu
    Real _d0;                     // rest density, rho0
    Real _h;                      // particle spacing
    Vec2f _g;                     // gravity

    Real _m0;                     // constant mass
    Real _p0;                     // equation of state (EOS) coefficient, k

// auxiliary variables; refer to [Mecker and Teschner, 2007, SCA]
    Real _eta;                    // allowed compression
    Real _c;                      // artificial speed of sound
    Real _gamma;                  // EOS power factor (typically use 7)
};

SphSolver gSolver(0.08, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp() {
    std::cout <<
              "> Help:" << std::endl <<
              "    Keyboard commands:" << std::endl <<
              "    * H: print this help" << std::endl <<
              "    * P: toggle simulation" << std::endl <<
              "    * G: toggle grid rendering" << std::endl <<
              "    * V: toggle velocity rendering" << std::endl <<
              "    * S: save current frame into a file" << std::endl <<
              "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height) {
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && key == GLFW_KEY_H) {
        printHelp();
    } else if (action == GLFW_PRESS && key == GLFW_KEY_S) {
        gSaveFile = !gSaveFile;
    } else if (action == GLFW_PRESS && key == GLFW_KEY_G) {
        gShowGrid = !gShowGrid;
    } else if (action == GLFW_PRESS && key == GLFW_KEY_V) {
        gShowVel = !gShowVel;
    } else if (action == GLFW_PRESS && key == GLFW_KEY_P) {
        gAppTimerStoppedP = !gAppTimerStoppedP;
        if (!gAppTimerStoppedP)
            gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
    } else if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
        glfwSetWindowShouldClose(window, true);
    }
}

void initGLFW() {
    // Initialize GLFW, the library responsible for window management
    if (!glfwInit()) {
        std::cerr << "ERROR: Failed to init GLFW" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Before creating the window, set some option flags
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    // Create the window
    gWindowWidth = gSolver.resX() * kViewScale;
    gWindowHeight = gSolver.resY() * kViewScale;
    gWindow = glfwCreateWindow(
            gSolver.resX() * kViewScale, gSolver.resY() * kViewScale,
            "Basic SPH Simulator", nullptr, nullptr);
    if (!gWindow) {
        std::cerr << "ERROR: Failed to open window" << std::endl;
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // Load the OpenGL context in the GLFW window
    glfwMakeContextCurrent(gWindow);

    // not mandatory for all, but MacOS X
    glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

    // Connect the callbacks for interactive control
    glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
    glfwSetKeyCallback(gWindow, keyCallback);

    std::cout << "Window created: " <<
              gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void initOpenGL() {
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init() {
    gSolver.initScene(48, 32, 16, 16);

    initGLFW();                   // Windowing system
    initOpenGL();
}

void clear() {
    glfwDestroyWindow(gWindow);
    glfwTerminate();
}

// The main rendering call
void render() {
    glClearColor(.4f, .4f, .4f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // grid guides
    if (gShowGrid) {
        glBegin(GL_LINES);
        for (int i = 1; i < gSolver.resX(); ++i) {
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(i), 0.0);
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
        }
        for (int j = 1; j < gSolver.resY(); ++j) {
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(0.0, static_cast<Real>(j));
            glColor3f(0.3, 0.3, 0.3);
            glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
        }
        glEnd();
    }

    // render particles
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glPointSize(0.25f * kViewScale);

    glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
    glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
    glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    // velocity
    if (gShowVel) {
        glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

        glEnableClientState(GL_VERTEX_ARRAY);

        glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
        glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);

        glDisableClientState(GL_VERTEX_ARRAY);
    }

    if (gSaveFile) {
        std::stringstream fpath;
        fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

        std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
        const short int w = gWindowWidth;
        const short int h = gWindowHeight;
        std::vector<int> buf(w * h * 3, 0);
        glReadPixels(0, 0, w, h, GL_BGR_EXT, GL_UNSIGNED_BYTE, &(buf[0]));

        FILE *out = fopen(fpath.str().c_str(), "wb");
        short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
        fwrite(&TGAhead, sizeof(TGAhead), 1, out);
        fwrite(&(buf[0]), 3 * w * h, 1, out);
        fclose(out);
        gSaveFile = false;

        std::cout << "Done" << std::endl;
    }
}

// Update any accessible variable based on the current time
void update(const float currentTime) {
    if (!gAppTimerStoppedP) {
        // Animate any entity of the program here
        const float dt = currentTime - gAppTimerLastClockTime;
        gAppTimerLastClockTime = currentTime;
        gAppTimer += dt;
        // <---- Update here what needs to be animated over time ---->

        // solve 10 steps
        for (int i = 0; i < 10; ++i) gSolver.update_spisph();
    }
}

int main() {
    init();
    while (!glfwWindowShouldClose(gWindow)) {
        update(static_cast<float>(glfwGetTime()));
        render();
        glfwSwapBuffers(gWindow);
        glfwPollEvents();
    }
    clear();
    std::cout << " > Quit" << std::endl;
    return EXIT_SUCCESS;
}

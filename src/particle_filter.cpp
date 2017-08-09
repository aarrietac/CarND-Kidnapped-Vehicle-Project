/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Add random Gaussian noise to each state of particles
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // set number of particles
    num_particles = 200;

    // set weight of particles to 1
    weights.resize(num_particles, 1.0f);

    // initialize each particle and add Gaussian noise
    for(int i = 0; i < num_particles; i++){
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0f;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // default normal distribution for sensor noise
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++){

        // extract states
        double px = particles[i].x;
        double py = particles[i].y;
        double theta = particles[i].theta;
        double v = velocity;
        double dtheta = yaw_rate;

        // predicted states using motion model
        double p_px, p_py, p_theta;

        // apply kinematic model
        if (fabs(dtheta < 0.01)){ // zero yaw rate
            p_px = px + v * cos(theta) * delta_t;
            p_py = py + v * sin(theta) * delta_t;
        }
        else{
            p_px = px + (v/dtheta)*(sin(theta + dtheta*delta_t) - sin(theta));
            p_py = py + (v/dtheta)*(cos(theta) - cos(theta + dtheta*delta_t));
        }

        p_theta = theta + dtheta*delta_t;

        // update states add noise for each particle state
        particles[i].x = p_px + dist_x(gen);
        particles[i].y = p_py + dist_y(gen);
        particles[i].theta = p_theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    // define some variables
    double min_distance;
    int min_index;

    // compute min distance to each landmark for each observation
    for (int i = 0; i < observations.size(); i++){
        auto observation = observations[i];

        min_distance = INFINITY;  // large value in the beginning
        min_index = -1;

        for (int j = 0; j < predicted.size(); j++){
            auto predict = predicted[j];
            double dx = predict.x - observation.x;
            double dy = predict.y - observation.y;
            double d  = dx*dx + dy*dy;
            if(d < min_distance){
                min_distance = d;
                min_index = j;
            }
        }
        observations[i].id = min_index;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // for each particle
    for (int i = 0; i < num_particles; i++){
        // get particle states
        double px     = particles[i].x;
        double py     = particles[i].y;
        double ptheta = particles[i].theta;

        // landmark within the range of the sensor (sensor specifications)
        vector<LandmarkObs> predictions;

        // for each landmark get predictions
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            // get landmark parameters
            float lxf = map_landmarks.landmark_list[j].x_f;
            float lyf = map_landmarks.landmark_list[j].y_f;
            int   lid = map_landmarks.landmark_list[j].id_i;

            // consider a circular one
            if ((lxf - px)*(lxf - px) + (lyf - py)*(lyf - py) <= sensor_range*sensor_range) {
                predictions.push_back(LandmarkObs{lid, lxf, lyf});
            }
        }

        // observation from vehicle coordinate system to map coordinate system
        vector<LandmarkObs> transformed_obs;
        for (int j = 0; j < observations.size(); j++) {
            double tx = px + cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y;
            double ty = py + sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y;
            int tid = observations[j].id;
            transformed_obs.push_back(LandmarkObs{tid, tx, ty});
        }

        // data association for the predictions and tranformed observations
        dataAssociation(predictions, transformed_obs);

        double new_weight = 1.0f;  // set new weights for particles
        for (int j = 0; j < transformed_obs.size(); j++) {

            auto obs = transformed_obs[j];
            auto assoc_lm = predictions[obs.id];

            // compute the weight for each the current observations
            double pdf = gaussian2D(obs, assoc_lm, std_landmark);
            new_weight *= pdf;
        }
        particles[i].weight = new_weight;
        weights[i] = new_weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // use discrete_distribution from source
    std::discrete_distribution<int> _disc(weights.begin(), weights.end());

    // vector of new particles
    vector<Particle> new_particles;

    for(int i = 0; i < num_particles; i++) {
        auto index = _disc(gen);
        new_particles.push_back(std::move(particles[index]));
    }
    particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

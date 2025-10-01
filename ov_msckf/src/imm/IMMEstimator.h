
//Implements an Interacting Multiple-Model (IMM) estimator.

    // Parameters
    // ----------

    // filters : (N,) array_like of KalmanFilter objects
    //     List of N filters. filters[i] is the ith Kalman filter in the
    //     IMM estimator.

    //     Each filter must have the same dimension for the state `x` and `P`,
    //     otherwise the states of each filter cannot be mixed with each other.

    // mu : (N,) array_like of float
    //     mode probability: mu[i] is the probability that
    //     filter i is the correct one.

    // M : (N, N) ndarray of float
    //     Markov chain transition matrix. M[i,j] is the probability of
    //     switching from filter j to filter i.


    // Attributes
    // ----------
    // x : numpy.array(dim_x, 1)
    //     Current state estimate. Any call to update() or predict() updates
    //     this variable.

    // P : numpy.array(dim_x, dim_x)
    //     Current state covariance matrix. Any call to update() or predict()
    //     updates this variable.

    // x_prior : numpy.array(dim_x, 1)
    //     Prior (predicted) state estimate. The *_prior and *_post attributes
    //     are for convienence; they store the  prior and posterior of the
    //     current epoch. Read Only.

    // P_prior : numpy.array(dim_x, dim_x)
    //     Prior (predicted) state covariance matrix. Read Only.

    // x_post : numpy.array(dim_x, 1)
    //     Posterior (updated) state estimate. Read Only.

    // P_post : numpy.array(dim_x, dim_x)
    //     Posterior (updated) state covariance matrix. Read Only.

    // N : int
    //     number of filters in the filter bank

    // mu : (N,) ndarray of float
    //     mode probability: mu[i] is the probability that
    //     filter i is the correct one.

    // M : (N, N) ndarray of float
    //     Markov chain transition matrix. M[i,j] is the probability of
    //     switching from filter j to filter i.

    // cbar : (N,) ndarray of float
    //     Total probability, after interaction, that the target is in state j.
    //     We use it as the # normalization constant.

    // likelihood: (N,) ndarray of float
    //     Likelihood of each individual filter's last measurement.

    // omega : (N, N) ndarray of float
    //     Mixing probabilitity - omega[i, j] is the probabilility of mixing
    //     the state of filter i into filter j. Perhaps more understandably,
    //     it weights the states of each filter by:
    //         x_j = sum(omega[i,j] * x_i)

    //     with a similar weighting for P_j

#pragma once

#include <memory>
#include <vector>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <limits>

namespace ov_msckf {
    class State;
    class IMMEstimator {
        public:
            IMMEstimator();
            void printDebugInfo() const;
            // double likelihood(const Eigen::MatrixXd& S, const Eigen::VectorXd& res) const;
            double likelihood(const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sinv, const Eigen::VectorXd& res);
            void record_likelihood(int model_id, double likelihood);
            void updateModelProbabilities(double timestamp);
            void saveModeProbHistoryCSV(const std::string& filename) const;
            std::pair<Eigen::MatrixXd, Eigen::VectorXd> mixingProbabilities();
            // std::vector<Eigen::VectorXd> mode_prob_history;
            std::vector<std::pair<double, Eigen::VectorXd>> mode_prob_history;
            std::shared_ptr<State> state;
            std::shared_ptr<State> state_1;
            
            
    
        private:
            int num_models;
            Eigen::VectorXd mu;
            Eigen::MatrixXd M;
            Eigen::VectorXd likelihoods;
    
    
    };
} // namespace ov_msckf



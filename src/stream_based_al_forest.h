// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 or the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2016
 * Dep. Of Computer Science
 * Technical University of Munich (TUM)
 *
 */

#ifndef STREAM_BASED_AL_FOREST_H_
#define STREAM_BASED_AL_FOREST_H_

/* Evaluate assertion */
//#define NDEBUG /* Comment out if no assertion is needed */
#include <assert.h> 

#include <list>
#include <algorithm>  /* Used for count elements in vector */
#include <armadillo>  /* Matrix, vector library */
#include "stream_based_al_random.h"
#include "stream_based_al_data.h"
#include "stream_based_al_tree.hpp"
#include "stream_based_al_hyperparameters.h"
#include "stream_based_al_metrics.hpp"
#include <limits>

/* Boost */
#include <boost/progress.hpp>
#include <boost/timer.hpp>


/*---------------------------------------------------------------------------*/
// Forward declaration of external random number generator
extern RandomGenerator rng;

/*---------------------------------------------------------------------------*/
/**
 * Defines a Mondrian forest -> defined number of Mondrian trees
 */
class MondrianForest {
    public:
        /**
         * Construct mondrian tree
         */
        MondrianForest(const mondrian_settings& settings, 
                const int& features_dim);

        ~MondrianForest();
        /**
         * Train forest with a single data sample with the option to perform
         * a density update.
         *
         * @param sample            Data sample for update.
         * @param density_update    Update to density estimate of forest.
         */ 
        void train(Sample& sample, bool density_update = false);
    
        /**
         * Function trains a Mondrian forest
         *
         * Input parameter:
         *
         * @param dataset   : Training dataset
         * @param hp        : Hyperparameters
         *
         */
        void train(DataSet& dataset, Hyperparameters& hp);

        /**
         * Function trains a Mondrian forest in an active learning setting
         *
         * Input parameter:
         *
         * @param dataset   : Training dataset
         * @param hp        : Hyperparameters
         *
         */
        void train_active(DataSet& dataset, Hyperparameters& hp);
    
        /**
         * Predict class of current sample
         */
        int classify(Sample& sample);
    
        /**
         * Predict class and return confidence
         *
         * @param sample            Data sample for update.
         * @param density_update    Update to density estimate of forest.
         */
        pair<int, float> classify_confident(Sample& sample, bool density_update = false);
        /**
         * Function tests/evaluates a Mondrian forest
         *
         * @param dataset   : Testing dataset
         * @param hp        : Hyperparameters
         */
        void classify(DataSet& dataset, Result& pResult, Hyperparameters& hp, bool density_update = false);
    
        float get_data_counter() const;

        void print_info();
        
    private:
        float data_counter_;  /**< Count incoming data points */
        vector<MondrianTree*> trees_;  /**< Save all Mondrian trees */
        const mondrian_settings* settings_;  /**< Settings of a Mondrian forest */
        /*
         * Calculates probability of current sample
         * (returns probability of all classes)
         */
        arma::fvec predict_probability(Sample& sample,
                mondrian_confidence& m_conf, bool density_update = false);
        /*
         * Calculates confidence value
         */
        float confidence_prediction(arma::fvec& prediction,
                mondrian_confidence& m_conf);
        
};

#endif /* STREAM_BASED_AL_FOREST_H_ */

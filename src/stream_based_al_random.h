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

#ifndef STREAM_BASED_AL_RANDOM_H_
#define STREAM_BASED_AL_RANDOM_H_

/*
 * Used to generate random numbers
 */
#include <boost/random/exponential_distribution.hpp> 
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>  /**< Random generator mt19937 */
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <sys/time.h>

#include "stream_based_al_utilities.h"

using namespace std;


/*---------------------------------------------------------------------------*/
/**< Type of random generator */
typedef boost::mt11213b base_generator_type;  /* mt11213b (faster), mt19937 */


/*---------------------------------------------------------------------------*/
class RandomGenerator {

    public:

        RandomGenerator();

        void set_seed(unsigned int);
    
        float rand_uniform_distribution();
        /**
         * Generate value that is uniformly distributed between min and max
         */
        float rand_uniform_distribution(float min_value, float max_value);
        float rand_exp_distribution(float lambda);
        int rand_discrete_distribution(arma::fvec& scores);

    private:
        base_generator_type generator; // base random number generator

        static unsigned int init_seed();
        boost::uniform_real<float> uni_dist;
        boost::variate_generator<base_generator_type&,
            boost::uniform_real<float> > uni_gen;

};

#endif /* STREAM_BASED_AL__RANDOM_H_ */
/*---------------------------------------------------------------------------*/

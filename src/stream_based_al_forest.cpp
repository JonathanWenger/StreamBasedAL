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

#include "stream_based_al_forest.h"


/*---------------------------------------------------------------------------*/
/*
 * Construct Mondrian forest
 */
MondrianForest::MondrianForest(const mondrian_settings& settings, 
                const int& feature_dim) :
    data_counter_(0),
    settings_(&settings) {
    MondrianTree* tree = NULL;
    for (int n_tree = 0; n_tree < settings.num_trees; n_tree++) {
        tree = new MondrianTree(settings, feature_dim);
        trees_.push_back(tree);
    }
}

MondrianForest::~MondrianForest() {
    /* Clear memory space */
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        delete trees_[n_tree];
    }
}

/*
 * Update current data point
 */ 
void MondrianForest::update(Sample& sample) {
    data_counter_++;

    /* Update all trees with current sample */
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        trees_[n_tree]->update(sample);
    }
}

/*
 * Predict class of current class
 */ 
int MondrianForest::classify(Sample& sample) {
    /* Go through all trees and calculate probability */
    //float expo_param = 1.0;
    mondrian_confidence m_conf = {0,0,0};
    arma::fvec pred_prob = predict_probability(sample, m_conf);

    int pred_class = -1;  /* Predicted class of Mondrian forest */ 
    /* If all probabilies are the same -> return -2 */
    if (equal_elements(pred_prob)) {
        return -2;
    }

    float tmp_value = 0.;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > tmp_value) {
            tmp_value = pred_prob[i];
            pred_class = i;
        }
    }
    
    return pred_class;
}
/**
 * Predict class and return confidence
 */
pair<int, float> MondrianForest::classify_confident(Sample& sample) {
    pair<int, float> prediction (0, 0.0);
    
    /* Distance value that influence prediction */
    /* Init confidence values */
    mondrian_confidence m_conf;

    /* Go through all trees and calculate probability */
    arma::fvec pred_prob = predict_probability(sample, m_conf);

    int pred_class = -1;  /* Predicted class of Mondrian forest */
    float tmp_value = 0.;
    for (int i = 0; i < int(pred_prob.size()); i++) {
        if (pred_prob[i] > tmp_value) {
            tmp_value = pred_prob[i];
            pred_class = i;
        }
    }
    prediction.first = pred_class;

    /* Calculate confidence */
    float confidence = confidence_prediction(pred_prob, m_conf);
    prediction.second = confidence;
    return prediction;
}

void MondrianForest::print_info() {
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        trees_[n_tree]->print_info();
    }
}

float MondrianForest::get_data_counter() const{
    return data_counter_;
}

/*
* Calculates probability of current sample
* (returns probability of all classes)
*/
arma::fvec MondrianForest::predict_probability(Sample& sample,
        mondrian_confidence& m_conf) {
    /* Go through all trees and calculate probability */
    arma::fvec pred_prob(trees_[0]->num_classes_, arma::fill::zeros);
    float tmp_normalized_density_forest = 0;
    for (int n_tree = 0; n_tree < settings_->num_trees; n_tree++) {
        arma::fvec tmp_pred_prob(trees_[0]->num_classes_, arma::fill::zeros);
        trees_[n_tree]->classify(sample, tmp_pred_prob, m_conf);
        tmp_normalized_density_forest += m_conf.normalized_density;
        pred_prob += tmp_pred_prob;
    }
    pred_prob = pred_prob / settings_->num_trees;
    m_conf.normalized_density = tmp_normalized_density_forest/settings_->num_trees;

    return pred_prob;
}


/*
* Calculates confidence value
*/
float MondrianForest::confidence_prediction(arma::fvec& pred_prob,
        mondrian_confidence& m_conf) {
    float uncertainty = 0.0;
    
    if(settings_->confidence_measure == 0){
    /* Confidence: first best vs. second best */
        float first_class = max(pred_prob);
        float second_class = 0.0;
        for (int i = 0; i < int(pred_prob.size()); i++) {
            if (pred_prob[i] > second_class && pred_prob[i] < first_class) {
                second_class = pred_prob[i];
            }
        }
        uncertainty = 1 - first_class + second_class;
    }
    else if(settings_->confidence_measure == 1){
    /* Confidence: normalized entropy */
        assert(pred_prob.size() > 1);
        for (int i = 0; i < pred_prob.size(); i++){
            if(pred_prob(i) > 0)
                uncertainty += -pred_prob(i)*log(pred_prob(i))/log(pred_prob.size());
        }
    }else if(settings_->confidence_measure == 2){
        uncertainty = m_conf.normalized_density;
    }else if(settings_->confidence_measure == 3){
        uncertainty = rng.rand_uniform_distribution(0, 1);
    }

    float beta = settings_->density_exponent;
    float confidence = 1 - uncertainty * pow((m_conf.normalized_density),beta);
    
    return confidence;
}

/**
 * Function trains a mondrian forest
 */
void MondrianForest::train(DataSet& dataset,  Hyperparameters& hp) {
    
    /* Set number of training samples */
    unsigned int number_training_samples = 0;
    if (hp.number_of_samples_for_training_ == 0)
        number_training_samples = dataset.num_samples_;
    else
        number_training_samples = hp.number_of_samples_for_training_;
    
    cout << endl;
    cout << "------------------" << endl;
    cout << "Start training ..." << endl;
    cout << "------------------" << endl;
    
    /* Check if test file exists */
    if (dataset.num_samples_ < 1) {
        cout << "[ERROR] - There does not exist a training dataset" << endl;
        exit(EXIT_FAILURE);
    }
    /* Initialize progress bar */
    unsigned int expected_count = dataset.num_samples_;
    /* Display training progress */
    boost::progress_display show_progress( expected_count );
    
    /* Initialize stop time for training */
    timeval startTime;
    gettimeofday(&startTime, NULL);
    
    /*---------------------------------------------------------------------*/
    /* Go through complete training set */
    int long i_samp = 0;
    //for (; i_samp < hp.number_of_samples_for_training_; i_samp++) {
    for (; i_samp < number_training_samples; i_samp++) {
        Sample sample = dataset.get_next_sample();
        update(sample);
        pResult_->samples_used_for_training_++;
        /* Show progress */
        ++show_progress;
    }
    
    
    /*---------------------------------------------------------------------*/
    cout << endl;
    cout << " ... finished training after: ";
    timeval endTime;
    gettimeofday(&endTime, NULL);
    float tmp_training_time = (endTime.tv_sec - startTime.tv_sec +
                               (endTime.tv_usec - startTime.tv_usec) / 1e6);
    cout << tmp_training_time << " seconds." << endl;
    
    pResult_->training_time_ += tmp_training_time;
    
}


/**
 * Function trains a mondrian forest in an active learning setting
 */
void MondrianForest::train_active(DataSet& dataset, Hyperparameters& hp) {
    
    /* Set number of training samples */
    unsigned int number_training_samples = 0;
    if (hp.number_of_samples_for_training_ == 0)
        number_training_samples = (int) dataset.num_samples_;
    else
        number_training_samples = hp.number_of_samples_for_training_;
    
    cout << endl;
    cout << "-------------------------------------" << endl;
    cout << "Start training (active learning " <<
    hp.active_learning_ << ")..." << endl;
    cout << "-------------------------------------" << endl;
    
    /* Check if test file exists */
    if (dataset.num_samples_ < 1) {
        cout << "[ERROR] - There does not exist a training dataset" << endl;
        exit(EXIT_FAILURE);
    }
    /* Initialize progress bar */
    unsigned int expected_count = dataset.num_samples_;
    /* Display training progress */
    boost::progress_display show_progress( expected_count );
    
    /* Initialize stop time for training */
    timeval startTime;
    gettimeofday(&startTime, NULL);
    
    /* Variables of active learning */
    vector<float> active_conf_values;
    
    /*---------------------------------------------------------------------*/
    /* Go through complete training set */
    
    /**
     * Options active learning:
     *  - 1 = active learning: updates mf with samples that are less than
     *                         "active_confidence_value_"
     *  - 2 = active learning: use only "active_buffer_percentage" samples
     *                         of the training set to update mf
     */
    if (hp.active_learning_ == 1) {
        for (int long i_samp = 0; i_samp < number_training_samples ; i_samp++) {
            Sample sample = dataset.get_next_sample();
            
            if (get_data_counter() < hp.active_init_set_size_) {
                /* Initial training set without active learning */
                update(sample);
                pResult_ -> samples_used_for_training_++;
            } else {
                /* Stop training if the number of samples used for training is larger than specified */
                if (pResult_->samples_used_for_training_ == hp.active_max_num_queries_){
                    break;
                }
                pair<int, float> pred = classify_confident(sample);
                if (pred.second < hp.active_confidence_value_) {
                    update(sample);
                    pResult_ -> samples_used_for_training_++;
                }
            }
            /* Show progress */
            ++show_progress;
        }
    } else if (hp.active_learning_ == 2) {
        
        /* Active learning with buffering samples to learn only samples that are very
         * uncertain (last x%)*/
        
        pair<Sample, float> i_active_sample;
        list<pair<Sample, float> > active_buffer;
        int count_buffer = 0;
        
        for (int long i_samp = 0; i_samp < number_training_samples; i_samp++) {
            Sample sample = dataset.get_next_sample();
            
            if (get_data_counter() < hp.active_init_set_size_) {
                update(sample);
                pResult_ -> samples_used_for_training_++;
            } else {
                /* Stop training if the number of samples used for training is larger than specified */
                if (pResult_->samples_used_for_training_ == hp.active_max_num_queries_){
                    break;
                }
                
                pair<int, float> pred = classify_confident(sample);
                i_active_sample.first = sample;
                i_active_sample.second = pred.second;
                /* Insert sample */
                insert_sort(active_buffer, i_active_sample);
                count_buffer++;
                
                if (count_buffer >= hp.active_batch_size_) {
                    /* Go through active buffer and update "active_buffer" of most uncertain
                     * samples */
                    list<pair<Sample, float> >::iterator it = active_buffer.begin();
                    for (int i_buf = 0; it != active_buffer.end(); it++) {
                        update((*it).first);
                        pResult_ -> samples_used_for_training_++;
                        if (i_buf == 0)
                            active_conf_values.push_back((*it).second);
                        if (i_buf == hp.active_buffer_size_){
                            active_conf_values.push_back((*it).second);
                            break;
                        }
                        ++i_buf;
                    }
                    count_buffer = 0;
                    active_buffer.clear();
                }
            }
            /* Show progress */
            ++show_progress;
        }
        
        
    } else {
        cout << "[Error: option for active learning is not available]" << endl;
        exit(EXIT_FAILURE);
    }
    
    /*---------------------------------------------------------------------*/
    cout << endl;
    cout << " ... finished training after: ";
    timeval endTime;
    gettimeofday(&endTime, NULL);
    float tmp_training_time = (endTime.tv_sec - startTime.tv_sec +
                               (endTime.tv_usec - startTime.tv_usec) / 1e6);
    cout << tmp_training_time << " seconds." << endl;
    
    pResult_ -> training_time_ += tmp_training_time;
}
/**
 * Function tests/evaluates a mondrian forest
 */
double MondrianForest::classify(DataSet& dataset, Hyperparameters& hp) {
    
    cout << endl;
    cout << "-----------------" << endl;
    cout << "Start testing ..." << endl;
    cout << "-----------------" << endl;
    cout << endl;
    
    /* Check if test file exists */
    if (dataset.num_samples_ < 1) {
        cout << "[ERROR] - There does not exist a test dataset." << endl;
        exit(EXIT_FAILURE);
    }
    
    /* Initialize stop time for training */
    timeval startTime;
    gettimeofday(&startTime, NULL);
    
    
    /* Initialize progress bar */
    unsigned int expected_count = dataset.num_samples_;
    /* Display training progress */
    boost::progress_display show_progress( expected_count );
    
    /*---------------------------------------------------------------------*/
    
    int pred_class = 0;  /* Predicted class */
    int conf_pos = 0;  /* Position of confidence value */
    
    /* Go through complete test set */
    for (unsigned int n_elem = 0; n_elem < dataset.num_samples_; n_elem++) {
        
        /* Get next sample */
        Sample sample = dataset.get_next_sample();
        
        pred_class = 0;
        
        if (conf_value_) {
            /*
             * Calculates a confidence value for each prediction and saves
             * it in some kind of bar representation for further visualization
             */
            pair<int, float> pred = mf->classify_confident(sample);
            pred_class = pred.first;
            conf_pos = int((pred.second * 100) / 5);
            
            if (conf_pos <= 20) { //TODO: understand and fix this
                if (conf_pos == 20)
                    conf_pos = 19;
                if (pred_class == sample.y) {
                    pResult_ -> confidence_[conf_pos] += 1;
                } else {
                    pResult_ -> confidence_false_[conf_pos] += 1;
                }
            } else {
                std::cout << "Warning: confidence value is wrong! " << conf_pos <<
                std::endl;
            }
        } else {
            /* Prediction */
            pred_class = mf->classify(sample);
        }
        
        pResult_ -> result_prediction_.push_back(pred_class);
        
        /* Show progress */
        ++show_progress;
    }
    
    /*---------------------------------------------------------------------*/
    
    cout << endl;
    cout << " ... finished testing after: ";
    timeval endTime;
    gettimeofday(&endTime, NULL);
    float tmp_testing_time = (endTime.tv_sec - startTime.tv_sec +
                              (endTime.tv_usec - startTime.tv_usec) / 1e6);
    pResult_ -> testing_time_ += tmp_testing_time;
    cout << tmp_testing_time << " seconds." << endl;
    
    /* Evaluate test results */
    compute_metrics(dataset, pResult);
    
    return pResult_ -> accuracy_;
}

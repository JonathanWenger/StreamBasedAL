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
#include <sys/time.h>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <iomanip>
/* Armadillo */
#include <armadillo>
/* Boost */
#include <boost/progress.hpp>
#include <boost/timer.hpp>
/* Mondrian libraries */
#include "stream_based_al_forest.h"
#include "stream_based_al_data.h"
#include "stream_based_al_hyperparameters.h"
#include "stream_based_al_experimenter.h"

/*
 * Help function
 */
void help() {
    cout << endl;
    cout << "Help function of StreamBased_AL: " << endl;
    cout << "Input arguments: " << endl;
    cout << "\t -h | -- help: \t will display help message." << endl;
    cout << "\t -c : \t\t path to the config file." << endl << endl;
    cout << "\t --train : \t Train the classifier." << endl;
    cout << "\t --test  : \t Test the classifier." << endl;
    cout << "\t --confidence: \t Calculates a confidence value for each prediction \n \t\t\t (works but will not be saved in some file)" << endl;
    cout << "\tExamples:" << endl;
    cout << "\t ./StreamBasedAL_MF -c conf/stream_based_al.conf --train --test" << endl;
}

int main(int argc, char *argv[]) {
    cout << endl;
    cout << "################" << endl;
    cout << "StreamBased_AL: " << endl;
    cout << "################" << endl;
    cout << endl;
    /* Program parameters */
    bool training = false, testing = false, conf_value = false;
/*---------------------------------------------------------------------------*/
    /*
     * Reading input parameters
     * ------------------------
     */
    /* Check if input arguments are specified */
    if(argc == 1) {
        cout << "\tNo input argument specified: aborting..." << endl;
        help();
        exit(EXIT_SUCCESS);
    }
    int input_count = 1;
    string conf_file_name;
    // Parsing command line
    while (input_count < argc) {
        if (!strcmp(argv[input_count], "-h") || !strcmp(argv[input_count], 
                    "--help")) {
            help();
            return EXIT_SUCCESS;
        } else if (!strcmp(argv[input_count], "-c")) {
            conf_file_name = argv[++input_count];
        } else if (!strcmp(argv[input_count], "--train")) {
            training = true;
        } else if (!strcmp(argv[input_count], "--test")) {
            testing = true;
        } else if (!strcmp(argv[input_count], "--confidence")) {
            conf_value = true;
        } else {
            cout << "\tUnknown input argument: " << argv[input_count];
            cout << ", please try --help for more information." << endl;
            help();
            exit(EXIT_FAILURE);
        }
        input_count++;
    }
    if (conf_file_name.length() < 1) {
        cout << "[ERROR] - No config file selected ... " << endl;
        help();
        exit(EXIT_FAILURE);
    }
   
/*---------------------------------------------------------------------------*/
    /*
     * Loading training data and get properties of data set
     * ----------------------------------------------------
     */
    /* Load hyperparameters of Mondrian forest */
    Hyperparameters hp(conf_file_name);
    
    /* Set the seed of the random number generator*/
    if(hp.user_seed_config_ != 0){
        rng.set_seed(hp.user_seed_config_);
    }
    
    cout << endl;
    cout << "------------------" << endl;
    cout << "Loading files  ..." << endl;
    cout << "------------------" << endl;
    /* Load training and testing data */
    DataSet dataset_train(hp.random_, hp.sort_data_, hp.iterative_);
    DataSet dataset_test;
    dataset_train.load(hp.train_data_, hp.train_labels_);
    dataset_test.load(hp.test_data_, hp.test_labels_);
    /* Set feature dimension */
    int feat_dim = dataset_train.feature_dim_;
    
    /*
     * Set settings of Mondrian forest
     * ----------------------------------------------------
     */
    mondrian_settings* settings = new mondrian_settings;
    settings->num_trees = hp.num_trees_; 
    settings->discount_factor = hp.discount_factor_;
    settings->decision_prior_hyperparam = hp.decision_prior_hyperparam_;
    settings->discount_param = settings->discount_factor * float(feat_dim);
    settings->debug = hp.debug_;
    settings->max_samples_in_one_node = hp.max_samples_in_one_node_;
    settings->confidence_measure = hp.confidence_measure_;
    settings->density_exponent = hp.density_exponent_;
    
    
/*---------------------------------------------------------------------------*/
    
    /* Initialize result vector */
    const int num_query_steps = hp.active_num_query_steps_;
    int max_num_queries = hp.active_max_num_queries_;
    Result result_arr[hp.num_runs_][num_query_steps];
    int samples_used_for_training[hp.num_runs_][num_query_steps];
    
    for (int i = 0; i < hp.num_runs_; i++){
        cout << endl;
        cout << "-------------------- Run " << i + 1 << "/";
        cout << hp.num_runs_ << " -----------------------" << endl;
        
        for (int j = 0; j < num_query_steps; j++){
            hp.active_max_num_queries_ = ((float)max_num_queries*(j+1))/num_query_steps;
            
            /* Initialize Mondrian forest */
            MondrianForest* forest = new MondrianForest(*settings, feat_dim);
            
            /* Initialize result */
            result_arr[i][j] = Result();
            samples_used_for_training[i][j] = forest->get_data_counter();
            
            if (training) {
              /* Option between active learning and without */
              if (hp.active_learning_ > 0)
                forest->train_active(dataset_train, hp);
              else
                forest->train(dataset_train, hp);
            }
            // Compute the number of samples used for training
            samples_used_for_training[i][j] = forest->get_data_counter() - samples_used_for_training[i][j];
            
            if (testing) {
                dataset_test.reset_position();
                forest->classify(dataset_test, result_arr[i][j], hp);

                cout << endl;
                cout << "------------------" << endl;
                cout << "Properties:       " << endl;
                cout << "------------------" << endl;
                cout << "Accuracy: \t" << result_arr[i][j].accuracy_ << endl;
                cout << endl;
                cout << "Total samples used for training: "
                << samples_used_for_training[i][j] << endl;
                cout << endl;
            }
            
            dataset_train.reset_position();
            // Free space
            delete forest;
        }
    }
    
    /*
     *  Compute the average results of all runs and print them to stdout
     */
    if (hp.num_runs_ > 1){
        cout << "-------------------------------" << endl;
        cout << "   Average results (" << hp.num_runs_ << " runs):" << endl;
        cout << "-------------------------------" << endl;
        const char separator    = ' ';
        const int numWidth      = 12;
        
        cout << left << setw(numWidth) << setfill(separator) << "Samples:";
        cout << left << setw(numWidth) << setfill(separator) << "Accuracy:";
        cout << left << setw(numWidth) << setfill(separator) << "MicroPrec:";
        cout << left << setw(numWidth) << setfill(separator) << "MacroPrec:";
        cout << left << setw(numWidth) << setfill(separator) << "MicroRec:";
        cout << left << setw(numWidth) << setfill(separator) << "MacroRec:";
        cout << endl;

        arma::fvec avg_accuracy(num_query_steps, arma::fill::zeros);
        arma::fvec avg_samples_used_for_training(num_query_steps, arma::fill::zeros);
        arma::fvec avg_micro_precision(num_query_steps, arma::fill::zeros);
        arma::fvec avg_micro_recall(num_query_steps, arma::fill::zeros);
        arma::fvec avg_macro_precision(num_query_steps, arma::fill::zeros);
        arma::fvec avg_macro_recall(num_query_steps, arma::fill::zeros);
        arma::fmat avg_confusion_matrix(dataset_test.num_classes_, dataset_test.num_classes_, arma::fill::zeros);
        
        for (int j = 0; j < num_query_steps; j++){
            for (int i = 0; i < hp.num_runs_; i++){
                avg_accuracy(j) += result_arr[i][j].accuracy_/hp.num_runs_;
                avg_samples_used_for_training(j) +=
                    (float)samples_used_for_training[i][j]/hp.num_runs_;
                avg_micro_precision(j) += result_arr[i][j].micro_avg_precision_/hp.num_runs_;
                avg_micro_recall(j) += result_arr[i][j].micro_avg_recall_/hp.num_runs_;
                avg_macro_precision(j) += result_arr[i][j].macro_avg_precision_/hp.num_runs_;
                avg_macro_recall(j) += result_arr[i][j].macro_avg_recall_/hp.num_runs_;
                if(j == num_query_steps - 1){
                    avg_confusion_matrix += result_arr[i][j].confusion_matrix_/hp.num_runs_;
                }
            }
            
            cout << left << setw(numWidth) << setfill(separator) << avg_samples_used_for_training(j);
            cout << left << setw(numWidth) << setfill(separator) << avg_accuracy(j);
            cout << left << setw(numWidth) << setfill(separator) << avg_micro_precision(j);
            cout << left << setw(numWidth) << setfill(separator) << avg_macro_precision(j);
            cout << left << setw(numWidth) << setfill(separator) << avg_micro_recall(j);
            cout << left << setw(numWidth) << setfill(separator) << avg_macro_recall(j);
            cout << endl;
        }
        cout << endl;
        cout << "Average confusion matrix (predicted class vs. actual class):" << endl;
        cout << avg_confusion_matrix;
    }

/*---------------------------------------------------------------------------*/
    /*
     * Free Space
     */
    delete settings;

    return 0;
}



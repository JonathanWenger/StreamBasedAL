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

#include "stream_based_al_metrics.hpp"


void Metrics::evaluate_predictions(DataSet& dataset_test, Result& pResult){
        dataset_test.reset_position();
        pResult.true_positives_.zeros(dataset_test.num_classes_);
        pResult.false_positives_.zeros(dataset_test.num_classes_);
        pResult.false_negatives_.zeros(dataset_test.num_classes_);
        pResult.true_negatives_.zeros(dataset_test.num_classes_);
        
        for (unsigned int n_elem = 0; n_elem < pResult.result_prediction_.size();
             n_elem++) {
            
            Sample sample = dataset_test.get_next_sample();
            if (pResult.result_prediction_[n_elem] == sample.y) {
                pResult.result_correct_prediction_.push_back(1);
                pResult.true_positives_(sample.y) += 1;
                for(int i = 0; i < dataset_test.num_classes_; i++){
                    if(i != sample.y)
                        pResult.true_negatives_(i) += 1;
                }
            } else {
                pResult.result_correct_prediction_.push_back(0);
                pResult.false_positives_(pResult.result_prediction_[n_elem]) += 1;
                pResult.false_negatives_(sample.y) += 1;
            }
        }
}

void Metrics::precision_recall(DataSet& dataset_test, Result& pResult){
    pResult.precision_.zeros(dataset_test.num_classes_);
    pResult.recall_.zeros(dataset_test.num_classes_);
    
    // Compute precision and recall for all classes (1vsAll)
    for (int i = 0; i < dataset_test.num_classes_; i++){
        if((pResult.true_positives_(i) + pResult.false_positives_(i)) > 0)
            pResult.precision_(i) = pResult.true_positives_(i)
            /(pResult.true_positives_(i) + pResult.false_positives_(i));
        if (pResult.true_positives_(i) + pResult.false_negatives_(i) > 0)
            pResult.recall_(i) = pResult.true_positives_(i)
            /(pResult.true_positives_(i) + pResult.false_negatives_(i));
    }
};
void Metrics::micro_avg_precision(DataSet& dataset_test, Result& pResult){
    pResult.micro_avg_precision_ = arma::accu(pResult.true_positives_)/
    (arma::accu(pResult.true_positives_) + arma::accu(pResult.false_positives_));
};
void Metrics::macro_avg_precision(DataSet& dataset_test, Result& pResult){
    pResult.macro_avg_precision_ = arma::accu(pResult.precision_)/pResult.precision_.size();
};
void Metrics::micro_avg_recall(DataSet& dataset_test, Result& pResult){
    pResult.micro_avg_recall_ = arma::accu(pResult.true_positives_)/
    (arma::accu(pResult.true_positives_) + arma::accu(pResult.false_negatives_));
};
void Metrics::macro_avg_recall(DataSet& dataset_test, Result& pResult){
    pResult.macro_avg_recall_ = arma::accu(pResult.recall_)/pResult.precision_.size();
};
void Metrics::accuracy(DataSet& dataset_test, Result& pResult){
    pResult.accuracy_ = (float) arma::accu(pResult.true_positives_) / pResult. result_prediction_.size();
};
void Metrics::confusion_matrix(DataSet& dataset_test, Result& pResult){
    dataset_test.reset_position();
    pResult.confusion_matrix_.zeros(dataset_test.num_classes_, dataset_test.num_classes_);
    
    for (unsigned int n_elem = 0; n_elem < pResult.result_prediction_.size();
         n_elem++) {
        Sample sample = dataset_test.get_next_sample();
        pResult.confusion_matrix_(pResult.result_prediction_[n_elem], sample.y) += 1;
    }
    pResult.confusion_matrix_ = pResult.confusion_matrix_/pResult.result_prediction_.size();
};


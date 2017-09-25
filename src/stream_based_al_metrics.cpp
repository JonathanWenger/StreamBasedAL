//
//  stream_based_al_metrics.cpp
//  StreamBasedAL
//
//  Created by Jonathan Wenger on 25.09.17.
//  Copyright © 2017 Jonathan Wenger. All rights reserved.
//

#include "stream_based_al_metrics.hpp"


/**
 * Evaluate test results
 */
void Metrics::compute_metrics(DataSet& dataset_test, Result& pResult) {
    
    dataset_test.reset_position();
    
    // Initialize metrics
    pResult.true_positives_.zeros(dataset_test.num_classes_);
    pResult.false_positives_.zeros(dataset_test.num_classes_);
    pResult.false_negatives_.zeros(dataset_test.num_classes_);
    pResult.true_negatives_.zeros(dataset_test.num_classes_);
    pResult.precision_.zeros(dataset_test.num_classes_);
    pResult.recall_.zeros(dataset_test.num_classes_);
    pResult.confusion_matrix_.zeros(dataset_test.num_classes_, dataset_test.num_classes_);
    
    // Evaluate predictions, compute the confusion matrix and save TP, FP, TN and FN
    unsigned int same_elements = 0;
    for (unsigned int n_elem = 0; n_elem < pResult.result_prediction_.size();
         n_elem++) {
        
        Sample sample = dataset_test.get_next_sample();
        if (pResult. result_prediction_[n_elem] == sample.y) {
            same_elements++;
            pResult. result_correct_prediction_.push_back(1);
            pResult.true_positives_(sample.y) += 1;
            for(int i = 0; i < dataset_test.num_classes_; i++){
                if(i != sample.y)
                    pResult.true_negatives_(i) += 1;
            }
        } else {
            pResult. result_correct_prediction_.push_back(0);
            pResult.false_positives_(pResult. result_prediction_[n_elem]) += 1;
            pResult.false_negatives_(sample.y) += 1;
        }
        pResult.confusion_matrix_(pResult.result_prediction_[n_elem], sample.y) += 1;
    }
    pResult.confusion_matrix_ = pResult.confusion_matrix_/pResult.result_prediction_.size();
    
    // Compute precision and recall for all classes (1vsAll)
    for (int i = 0; i < dataset_test.num_classes_; i++){
        if((pResult.true_positives_(i) + pResult.false_positives_(i)) > 0)
            pResult.precision_(i) = pResult.true_positives_(i)
            /(pResult.true_positives_(i) + pResult.false_positives_(i));
        if (pResult.true_positives_(i) + pResult.false_negatives_(i) > 0)
            pResult.recall_(i) = pResult.true_positives_(i)
            /(pResult.true_positives_(i) + pResult.false_negatives_(i));
    }
    
    // Compute micro averages
    pResult.micro_avg_precision_ = arma::accu(pResult.true_positives_)/
    (arma::accu(pResult.true_positives_) + arma::accu(pResult.false_positives_));
    pResult.micro_avg_recall_ = arma::accu(pResult.true_positives_)/
    (arma::accu(pResult.true_positives_) + arma::accu(pResult.false_negatives_));
    
    // Compute macro averages
    pResult.macro_avg_precision_ = arma::accu(pResult.precision_)/pResult.precision_.size();
    pResult.macro_avg_recall_ = arma::accu(pResult.recall_)/pResult.precision_.size();
    
    // Compute accuracy
    if (same_elements != 0) {
        pResult.accuracy_ = (float) same_elements / pResult. result_prediction_.size();
    } else {
        pResult.accuracy_ = 0.0;
    }
}

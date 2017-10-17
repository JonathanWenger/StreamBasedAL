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

#ifndef stream_based_al_metrics_hpp
#define stream_based_al_metrics_hpp

#include "stream_based_al_data.h"

class Metrics{
    
public:
    static void evaluate_predictions(DataSet& dataset_test, Result& pResult);
    static void precision_recall(DataSet& dataset_test, Result& pResult);
    static void micro_avg_precision(DataSet& dataset_test, Result& pResult);
    static void macro_avg_precision(DataSet& dataset_test, Result& pResult);
    static void micro_avg_recall(DataSet& dataset_test, Result& pResult);
    static void macro_avg_recall(DataSet& dataset_test, Result& pResult);
    static void accuracy(DataSet& dataset_test, Result& pResult);
    static void confusion_matrix(DataSet& dataset_test, Result& pResult);
};

#endif /* stream_based_al_metrics_hpp */

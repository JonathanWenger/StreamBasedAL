//
//  stream_based_al_metrics.hpp
//  StreamBasedAL
//
//  Created by Jonathan Wenger on 25.09.17.
//  Copyright © 2017 Jonathan Wenger. All rights reserved.
//

#ifndef stream_based_al_metrics_hpp
#define stream_based_al_metrics_hpp

#include "stream_based_al_data.h"

class Metrics{
    
public:
    static void compute_metrics(DataSet& dataset_test, Result& pResult);
    
};

#endif /* stream_based_al_metrics_hpp */

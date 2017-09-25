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

#include "stream_based_al_experimenter.h"


/*
 * Construct experimenter
 */
Experimenter::Experimenter() :
  conf_value_(false) {
  pResult_ = new Result();
}
Experimenter::Experimenter(const bool confidence) :
  conf_value_(confidence) {
  pResult_ = new Result();
}
Experimenter::~Experimenter() {
  delete pResult_;
}

/**
 * Return training time
 */
double Experimenter::get_training_time() {
  return pResult_ -> training_time_;
}
/**
 * Return testing time
 */
double Experimenter::get_testing_time() {
  return pResult_ -> testing_time_;
}
/**
 * Return accuracy value
 */
double Experimenter::get_accuracy() {
  return pResult_ -> accuracy_;
}

/** 
 * Return detailed result
 */
Result Experimenter::get_detailed_result() {
  return (*pResult_);
}

package org.apache.spark.ml.made

import breeze.linalg.sum
import org.apache.spark.ml.linalg.{Vector, Vectors}


class Gradient extends Serializable {
  // Compute the gradient and loss
  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val _delta = sum(data.asBreeze *:* weights.asBreeze) - label // h_theta(x) - y_i
    val _loss = _delta * _delta / 2.0                               // diff^2 / 2
    val _grad = data.copy.asBreeze * _delta                   // \d / \dTheta J_Theta = x * _delta
    (Vectors.fromBreeze(_grad), _loss)
  }
}
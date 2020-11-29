package org.apache.spark.ml.made

import breeze.linalg.{Vector => BV, axpy => brzAxpy}
import org.apache.spark.ml.linalg.{Vector, Vectors}

// via https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/optimization/Updater.scala
abstract class Updater private[made] extends Serializable {
  def compute(
               weightsOld: Vector,
               gradient: Vector,
               stepSize: Double,
               iter: Int): Vector
}

class BasicUpdater private[made] extends Updater {
  override def compute(weightsOld: Vector,
                       gradient: Vector,
                       stepSize: Double,
                       iter: Int): Vector = {
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    // use specialized axpy for better performance
    brzAxpy(-stepSize, gradient.asBreeze, brzWeights)

    Vectors.fromBreeze(brzWeights)
  }
}
package org.apache.spark.ml.made

import breeze.linalg.norm
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.sql.expressions.Aggregator

import scala.collection.mutable.ArrayBuffer


class GradientDescent private[made] (private var gradient: Gradient,
                                     private var updater: Updater,
                                     private var inputCol: String,
                                     private var outputCol: String)
  extends Serializable with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100

  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def setInputCol(inputCol: String): this.type = {
    this.inputCol = inputCol
    this
  }

  def setOutputCol(outputCol: String): this.type = {
    this.outputCol= outputCol
    this
  }

  def check_norm(currWeights: Vector,
                 nextWeights: Vector): Boolean = {
    val _norm_val: Double = norm(currWeights.asBreeze.toDenseVector - nextWeights.asBreeze.toDenseVector)
    _norm_val < 0.001 * Math.max(norm(nextWeights.asBreeze.toDenseVector), 1.0)
  }

  def evaluate(dataset: Dataset[_], initialWeights: Vector): (Vector) = {

    var bestLoss = Double.MaxValue

    var currWeights: Option[Vector] = None
    var nextWeights: Option[Vector] = None

    var weights = Vectors.dense(initialWeights.toArray)
    var bestWeights = weights
    val weights_count = weights.size
    val rows_count = dataset.count()

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) { //!converged &&

      val customSummer =  new Aggregator[Row, (Vector, Double), (Vector, Double)] {
        def zero: (Vector, Double) = (Vectors.zeros(weights_count), 0.0)
        def reduce(acc: (Vector, Double), x: Row): (Vector, Double) = {
          val (grad, loss) = gradient.compute(x.getAs[Vector](inputCol), x.getAs[Double](outputCol), weights)
          (Vectors.fromBreeze(acc._1.asBreeze + grad.asBreeze / rows_count.asInstanceOf[Double]),
            acc._2 + loss / rows_count.asInstanceOf[Double])
        }
        def merge(acc1: (Vector, Double), acc2: (Vector, Double)): (Vector, Double) = (Vectors.fromBreeze(acc1._1.asBreeze + acc2._1.asBreeze), acc1._2 + acc2._2)
        def finish(r: (Vector, Double)): (Vector, Double) = r
        override def bufferEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()
        override def outputEncoder: Encoder[(Vector, Double)] = ExpressionEncoder()
      }.toColumn

      val row = dataset.select(customSummer.as[(Vector, Double)](ExpressionEncoder()))

      val loss = row.first()._2
      weights = updater.compute(weights, row.first()._1, stepSize, i)


      currWeights = nextWeights
      nextWeights = Some(weights)

      if (currWeights.isDefined && nextWeights.isDefined) {
        converged = check_norm(currWeights.get, nextWeights.get)
      }

      if (loss < bestLoss) {
        bestLoss = row.first()._2
        bestWeights = weights
      }

      println(s"Iter = ${i}, numIterations = ${numIterations}")
      i += 1
    }


    weights
  }


}

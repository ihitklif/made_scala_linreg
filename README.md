# made_scala_linreg

## Линейная регрессия Breeze + Spark ML

* Построить линейную регрессию на градиентном спуске

* Случайная матрица 100000x3
* «Скрытая модель» (1.5,0.3,-0.7)
* Решение основано на семинаре и https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/regression/LinearRegression.scala

* Тесты проверяют работу модели  на случайной матрице и сравнивают с моделью «из коробки»
* Также проведено тестирование сохранение модели и ее перезапуска

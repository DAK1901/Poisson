# Poisson
Решается задача Дирихле для уравнения Пуассона
-Δu = f
u|{∂Ω} = g
в сложной области Ω.
Область Ω имеется вид:
![image](https://github.com/DAK1901/Poisson/assets/141585517/b18cd450-13fd-4c7a-9361-1ef7b1d0cfba)
и разбита на подобласти (метод Шварца):
![photo_2023-10-21_19-13-29](https://github.com/DAK1901/Poisson/assets/141585517/46bc5478-cffd-4449-bb60-3dd79210ce35)
Для решения реализованы методы Якоби и алгоритмом неполной ортогонализации IOM (incomplete orthogonalization method).
Для визуализации решения использована библиотека matplotlib.pyplot.
Решение реализовано на языке python.

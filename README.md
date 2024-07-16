# ODE-Transformer
ODE neural network, based on encoder-decoder architecture and Autocorrelation attention mechanism
Модель для прогнозирования многомерных временных рядов на произвольный шаг. Разработана в рамках магистрской выпускной квалификационной работы.
Под сетями непрерывными по времени подразумевается ODE Net(https://arxiv.org/abs/1806.07366) в которой объединяются идеи нейронных сетей и обыкновенных дифференциальных уравнений. В данном подходе переход от 
слоя к слою нейронной сети, рассматривается как сложная трансформация её скрытого состояния:

$ℎ_{𝑡+1} = ℎ_{𝑡} + 𝑓(ℎ_{𝑡}, 𝜃_{𝑡})$

В пределе непрерывная динамика скрытого состояния параметризируется с помощью обыкновенного дифференциального уравнения (ODE), которое аппроксимируется нейронной сетью:

$\frac{𝑑ℎ(𝑡)}{𝑑𝑡} = 𝑓(ℎ(𝑡),𝑡, 𝜃)$

Скрытые состояния ℎ(𝑡) могут быть оценены в нужные моменты времени путем интегрирования функции ODE за определенный интервал времени с начальным значением. Таким образом можно моделировать динамику временного ряда делать предсказания с произвольным (в том числе и не дискретным) шагом как показано на рисунке:

![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202024-07-16%20134220.png)

На вход модель принимает многомерный временной ряд (целевая + независимые переменные). На выходе - предсказания целевой переменной на произвольные шаги. При сравнении с сущесвтующими моделями использовалось прогнозирование на шаги: [1, 1.5, 2, 2.5, 3].

# Арихеткутра разработанной моедли:

![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202024-07-16%20143843.png)

Модель основана на архитектуре Encoder-Decoder. В качестве Encoder части выстют: Модуль [Autocorrelation-attention](https://github.com/thuml/Autoformer) для обработки независимых переменных и оценки их влияния на целевую; Модуль ODE-LSTM (реккурентная сеть, скрытое состояние которой изменяется с помощью ODE) для обработки целевого ряда. Decoder - ODEnet где в качестве аппрокисирующей nn выступает LSTM.

![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/arch.png)
# Использование программы
Установка зависимостей ```pip install -r requirements.txt```

# Запуск скрипта
```python two_ode_selfatt.py -d datasetname```

# Результаты
Сравнение производилось на 4 датасетах:
* SML2010
* Electricity
* Electricity Transformer Temperature

Все они находятся в папке raw_data. Используемые метрики RMSE и MAE. Архитиктуры с которыми сравнивалась модель:
* [Latent ODE](https://github.com/YuliaRubanova/latent_ode)
* [Latent ODE-RNN](https://github.com/ashysheya/ODE-RNN)
* [ETN-ODE](https://github.com/PengleiGao/ETN-ODE)
* [EgPDE-Net](https://github.com/PengleiGao/EgPDE-net)

![RMSE](https://github.com/Dortp68/ODE-Transformer/blob/main/images/RMSE.png)
![MAE](https://github.com/Dortp68/ODE-Transformer/blob/main/images/MAE.png)


### Сравнение результатов на датасете SML2010
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/sml2010.png)
### Сравнение результатов на датасете ETTH1
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/ETTH1.png)
### Сравнение результатов на датасете ETTH2
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/ETTH2.png)
### Сравнение результатов на датасете Electricity
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/Electricity1.png)
### Результат предсказания моделей на шаге 1.5
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/timestep_1.5.png)
### Результат предсказания моделей на шаге 2.5
![](https://github.com/Dortp68/ODE-Transformer/blob/main/images/timestep_2.5.png)


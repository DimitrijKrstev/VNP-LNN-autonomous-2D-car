# ВНП - Liquid Neural Network за автономно движење на кола во 2Д простор

## Вовед во темата


Liquid Neural Networks (LNN) се нов тип невронски мрежи кои се особено ефикасни за обработка на податоци што постојано се менуваат. Тие се дизајнирани да бидат флексибилни и динамични, прилагодувајќи ги своите параметри во реално време. Ова ги прави идеални за апликации како роботика, автономни возила и биолошки инспирирани системи. LNN се помали и поинтерпретабилни од традиционалните мрежи и се робусни во бучни услови.

Reinforcement Learning (RL) е гранка на машинското учење каде агенти учат преку интеракција со околината, користејќи награди и казни за да ги оптимизираат своите одлуки. Наместо експлицитни инструкции, агентите истражуваат и учат преку проба и грешка. RL е посебно применлив во автономни системи, стратегии и игри.

Комбинацијата на LNN и RL е потенцијално корисна за динамични и непредвидливи средини, каде што моделите се тренираат во симулации и потоа применуваат во реални ситуации.

## Инсталација

За инсталирање и извршување на проектот се потребни следните команди:
```
git clone https://github.com/DimitrijKrstev/VNP-LNN-autonomous-2D-car.git  
cd VNP-LNN-autonomous-2D-car/ && pip install -r requirements.txt
```
Дополнително, за вршење на околината потребно е инсталирање на swig, препорачливо преку packet manager како apt, choco или brew.

## Околина за симулирање на автомобил

За околината каде што го тренираме нашиот агент да се движи се користи python библиотеката gymnasium, која обезбедува API за средини за reinforcement learning и вклучува нивни имплементации. Поточно, ја користиме box2d околината [CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/) за поедноставно симулирање на движење на кола во 2-димензионален простор.

Акцискиот простор за околината, која за овој проект е поставена да има дискретни акции, има 5 можни акции:

-   0: do nothing
-   1: steer left
-   2: steer right
-   3: gas   
-   4: brake

Додека пак, обзервациската околина ни дава RGB слика со 96x96 димензии што одпосле ја конвертираме во grayscale слика и ја користиме за влезот на нашата невронска мрежа. 

Наградата што ја дава околината е -0,1 секоја рамка и +1000/N за секоја посетена плочка на патеката, каде што N е вкупниот број на посетени плочки во патеката.

Агентот е завршен кога ќе ја помине секоја плочка на патеката или ќе искочи целосно надвор од границите на мапата.

# Time-series Forecasting: Are (Cross-)Attentions Necessary?

## 1. Wstęp
W prognozowaniu szeregów czasowych coraz częściej stosuje się modele oparte na mechanizmie attention, w szczególności architektury Transformer. Jednocześnie wyniki ostatnich prac pokazują, że proste modele liniowe mogą osiągać porównywalne lub lepsze rezultaty niż modele wykorzystujące self-attention.

Problemem badawczym jest ocena, czy mechanizm attention rzeczywiście poprawia jakość prognoz w długoterminowym prognozowaniu szeregów czasowych, czy też jego rola jest drugorzędna względem innych elementów architektury. W ramach zadania porównano modele z self-attention, modele bez attention oraz architekturę opartą wyłącznie na cross-attention, analizując ich skuteczność w jednolitym protokole eksperymentalnym.

## 2. Opis testowanych modeli

### 2.1 AutoARIMA (Baseline, Linear)
Model ARIMA, czyli AutoRegressive Integrated Moving Average, jest liniowym modelem prognostycznym dla szeregów czasowych, którego architektura opiera się na trzech połączonych mechanizmach. Pierwszym z nich jest różnicowanie danych w celu uzyskania stacjonarności, gdzie parametr d określa liczbę wykonanych operacji różnicowania. Drugim mechanizmem jest modelowanie zależności od przeszłych wartości szeregu w ramach części autoregresyjnej, przy czym parametr p oznacza liczbę opóźnień wykorzystywanych do opisu bieżącej obserwacji. Trzecim elementem jest korekta błędów prognozy realizowana przez mechanizm średniej ruchomej, w którym parametr q określa liczbę opóźnionych składników losowych, czyli reszt uwzględnianych w modelu. W pierwszym etapie działania modelu dane wejściowe są różnicowane w celu usunięcia trendu oraz ustabilizowania własności statystycznych szeregu. Następnie bieżąca wartość zmiennej jest opisywana jako liniowa kombinacja określonej liczby wcześniejszych obserwacji oraz wcześniejszych składników losowych. Prognozowanie odbywa się w sposób iteracyjny, przy czym kolejne wartości są wyznaczane na podstawie historii szeregu czasowego oraz własnych wcześniejszych prognoz modelu. W rozszerzeniu AutoARIMA do modelu wprowadzany jest dodatkowo komponent sezonowości, gdzie parametr s określa długość cyklu sezonowego. W tym wariancie wartości parametrów: d, p, q, s, modelu są wyznaczane automatycznie z wykorzystaniem testów statystycznych i kryteriów informacyjnych.

### 2.2 DLinear (Baseline, Linear)

Model DLinear jest prostym, liniowym modelem do długoterminowego prognozowania szeregów czasowych, którego działanie opiera się na jawnej dekompozycji danych wejściowych na komponent trendu oraz komponent sezonowy. W pierwszym kroku szereg czasowy jest rozdzielany przy użyciu średniej ruchomej na część trendową i część resztową reprezentującą sezonowość, a następnie każdy z tych komponentów jest przetwarzany niezależnie przez jednowarstwową transformację liniową działającą wzdłuż osi czasu. Ostateczna prognoza powstaje poprzez zsumowanie predykcji obu komponentów, co pozwala modelowi jednocześnie uchwycić długookresową dynamikę trendu oraz krótkookresowe wahania sezonowe. Dzięki bezpośredniemu wielokrokowemu prognozowaniu, braku mechanizmów autoregresyjnych i nieliniowych oraz bardzo niskiej złożoności obliczeniowej, DLinear stanowi interpretowalny i skuteczny model bazowy dla długoterminowego prognozowania szeregów czasowych.

### 2.3 Autoformer (self-attention)

Model Autoformer jest architekturą opartą na Transformerze, zaprojektowaną specjalnie do długoterminowego prognozowania szeregów czasowych, której działanie opiera się na połączeniu progresywnej dekompozycji sygnału oraz mechanizmu Auto-Correlation. W przeciwieństwie do klasycznych modeli self-attention, Autoformer wbudowuje dekompozycję szeregu czasowego bezpośrednio w strukturę sieci, dzięki czemu w trakcie prognozowania stopniowo oddziela komponent trendowo-cykliczny od komponentu sezonowego. Trend jest sukcesywnie akumulowany w dekoderze, natomiast sezonowość jest modelowana za pomocą nowego mechanizmu Auto-Correlation, który zastępuje klasyczną self-attention i identyfikuje zależności czasowe na poziomie całych podszeregów, wykorzystując okresowość danych. Mechanizm ten opiera się na obliczaniu autokorelacji sygnału i agregowaniu podobnych fragmentów szeregu przesuniętych w czasie, co pozwala na efektywne uchwycenie długookresowych zależności.

### 2.4 PatchTST (self-attention)

Model PatchTST jest architekturą opartą na klasycznym Transformerze, zaprojektowaną do długoterminowego prognozowania wielowymiarowych szeregów czasowych poprzez wprowadzenie dwóch kluczowych mechanizmów: patchingu oraz niezależności kanałów. W pierwszym etapie każda zmienna czasowa jest traktowana jako osobny, jednowymiarowy kanał i niezależnie normalizowana, a następnie dzielona na krótkie, lokalne podszeregi zwane patchami, które pełnią rolę tokenów wejściowych dla enkodera Transformera. Patche są następnie rzutowane do przestrzeni ukrytej, wzbogacane o osadzanie pozycyjne i przetwarzane przez standardowy enkoder Transformera z mechanizmem self-attention, przy czym te same wagi sieci są współdzielone pomiędzy wszystkimi kanałami. Ostateczna prognoza powstaje poprzez agregację reprezentacji uzyskanych dla poszczególnych kanałów i projekcję liniową do horyzontu predykcji.

### 2.5 iTransformer (self-attention)

Model iTransformer jest architekturą opartą na klasycznym Transformerze, która wprowadza odwrócenie sposobu reprezentacji danych w prognozowaniu wielowymiarowych szeregów czasowych. Zamiast traktować kolejne kroki czasowe jako tokeny zawierające wiele zmiennych, iTransformer interpretuje każdą zmienną czasową jako osobny token, do którego przypisana jest cała jej historia obserwacji. W takiej architekturze mechanizm self-attention jest stosowany na poziomie tokenów zmiennych i służy do modelowania korelacji między różnymi zmiennymi, natomiast sieć feed-forward jest wykorzystywana do uczenia reprezentacji czasowych każdej zmiennej niezależnie. Dane wejściowe są najpierw normalizowane i mapowane do przestrzeni ukrytej, następnie przetwarzane przez stos bloków Transformera bez modyfikacji jego podstawowych komponentów, a na końcu rzutowane liniowo do horyzontu prognozy.


### 2.6 Temporal Fusion Transformer (self-attention)

Model Temporal Fusion Transformer jest architekturą przeznaczoną do wielohoryzontowego prognozowania szeregów czasowych, która łączy mechanizmy sekwencyjne i attention w celu jednoczesnego modelowania zależności krótkoterminowych i długoterminowych. Na wejściu model rozróżnia zmienne statyczne, zmienne czasowe obserwowane w przeszłości oraz zmienne znane z wyprzedzeniem, a następnie wykorzystuje sieci selekcji zmiennych do dynamicznego wyboru najbardziej istotnych cech w każdym kroku czasowym. Informacja ze zmiennych statycznych jest kodowana w postaci wektorów kontekstu, które warunkują dalsze przetwarzanie sekwencji. Lokalne zależności czasowe są modelowane za pomocą warstwy sequence-to-sequence opartej na LSTM, natomiast zależności długookresowe są uchwytywane przez interpretable multi-head self-attention, umożliwiający analizę wpływu odległych punktów w czasie na prognozę. Cała architektura wykorzystuje bramkowane połączenia rezydualne, które pozwalają adaptacyjnie ograniczać złożoność modelu i tłumić nieistotne komponenty.

### 2.7 Cross-Attention-only Time Series Transformer (cross-attention)

Model CATS (Cross-Attention-only Time Series Transformer) jest architekturą prognostyczną zaprojektowaną specjalnie dla szeregów czasowych, która całkowicie rezygnuje z mechanizmu self-attention i opiera swoje działanie wyłącznie na mechanizmie cross-attention. Podstawową ideą modelu jest traktowanie każdej prognozowanej chwili w przyszłości jako osobnego zapytania, czyli learnable query zależnego od horyzontu prognozy, natomiast historyczny szereg czasowy pełni rolę kluczy i wartości. Dzięki temu prognoza dla każdego kroku czasowego w przyszłości jest obliczana niezależnie, jako wynik dopasowania konkretnego zapytania do przeszłych obserwacji. Architektura wykorzystuje patching danych wejściowych oraz szerokie współdzielenie parametrów pomiędzy różnymi horyzontami prognozy, co znacząco ogranicza liczbę parametrów i zużycie pamięci. Dodatkowo wprowadzony mechanizm query-adaptive masking losowo ogranicza dostęp do informacji historycznej, zmuszając model do silniejszego uwzględniania zapytań związanych z konkretnym horyzontem czasowym.

### Chronos (self-attention + cross-attention)

Model Chronos (w tym przypadku amazon/chronos-t5-small) jest wstępnie wytrenowanym modelem prognostycznym opartym na architekturze T5, zaprojektowanym do prognozowania szeregów czasowych w trybie zero-shot, bez potrzeby dodatkowego uczenia na danych zadaniowych. Jego działanie polega na traktowaniu szeregu czasowego jako sekwencji tokenów, które po odpowiedniej normalizacji i kwantyzacji są przekształcane do postaci analogicznej do danych tekstowych i przetwarzane przez enkoder–dekoder Transformera. Model uczy się generować przyszłe wartości autoregresyjnie, token po tokenie, korzystając wyłącznie z kontekstu historycznego dostarczonego na wejściu. Chronos jest trenowany na bardzo dużym i zróżnicowanym zbiorze rzeczywistych oraz syntetycznych szeregów czasowych, co pozwala mu internalizować ogólne wzorce trendów, sezonowości i zmienności.

## 3. Konfiguracja eksperymentów

Do ewaluacji modeli prognozowania szeregów czasowych wykorzystano zestaw danych pochodzący z benchmarku TSB-UAD, obejmujący rzeczywiste i przetworzone szeregi czasowe z różnych domen aplikacyjnych. Pomimo pierwotnego przeznaczenia benchmarku do detekcji anomalii, dane stanowią zróżnicowany i wymagający zbiór testowy dla zadań prognozowania, ze względu na obecność trendów, sezonowości, niestacjonarności oraz nagłych zmian dynamiki. W eksperymentach wykorzystano szeregi czasowe z następujących dziedzin:

- Environment-M – dane pochodzące z systemów monitoringu środowiskowego, obejmujące wielowymiarowe pomiary z czujników środowiskowych.
- Facility-U – jednowymiarowe szeregi czasowe pochodzące z systemów monitorujących funkcjonowanie infrastruktury technicznej i obiektów przemysłowych.
- Facility-M – wielowymiarowe dane czasowe pochodzące z infrastruktury technicznej, rejestrujące równoległe pomiary wielu powiązanych zmiennych.
- Finance-U – jednowymiarowe szeregi czasowe pochodzące z domeny finansowej, obejmujące dane rynkowe i transakcyjne.
- Medical-M – wielowymiarowe szeregi czasowe pochodzące z aplikacji medycznych, w tym z systemów monitorowania parametrów fizjologicznych.
- Webservices-U – jednowymiarowe dane czasowe pochodzące z systemów i usług internetowych, rejestrujące metryki pracy aplikacji i ruchu sieciowego.

Do oceny jakości prognoz szeregów czasowych wykorzystano następujące metryki:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

W przypadku danych wielowymiarowych (multivariate) metryki MAE, MSE, MAPE oraz MASE obliczano oddzielnie dla każdego wymiaru (kanału). Następnie raportowano dodatkowo:

- sumę wartości metryk wzdłuż wszystkich wymiarów,
- średnią arytmetyczną metryk obliczoną po wymiarach.

Dla każdego modelu dobrano optymalne wartości hiperparametrów z wykorzystaniem narzędzia Optuna.



## 4. Dobór hiperparametrów i optymalizacja

Proces strojenia hiperparametrów w zadaniu prognozowania długookresowego (**LTSF – Long-term Time Series Forecasting**) stanowi unikalne wyzwanie inżynieryjne, wykraczające poza standardowe procedury znane z klasycznego uczenia maszynowego. Wykorzystanie biblioteki **Optuna** wraz z algorytmiką **TPE (Tree-structured Parzen Estimator)** pozwoliło na inteligentne i probabilistyczne próbkowanie przestrzeni parametrów. Było to rozwiązanie krytyczne ze względu na drastycznie ograniczone zasoby czasowe oraz sprzętowe. 

W przeciwieństwie do naiwnego przeszukiwania siatki (*Grid Search*), algorytm TPE modeluje rozkład prawdopodobieństwa dobrych wyników poprzez podejście bayesowskie. Pozwoliło to systemowi na dynamiczne uczenie się nieliniowych relacji między parametrami a końcowym błędem kwadratowym (MSE) w trakcie trwania eksperymentu. Dzięki temu uniknięto marnowania cykli GPU na obszary przestrzeni, które nie rokowały poprawy metryk.



---

### Szczegółowa konfiguracja przestrzeni poszukiwań

Dla każdego z modeli zdefiniowano dedykowaną przestrzeń poszukiwań, dopasowaną do jego charakterystyki matematycznej oraz wymagań dotyczących stabilności gradientu:

* **CATS** (12 prób / trials)
    * `seq_len`: [48, 96, 192, 336, 512] – kluczowy parametr okna wejściowego decydujący o horyzoncie kontekstowym.
    * `n_layers`: [1, 2, 3] – głębokość sieci rzutująca na stopień abstrakcji cech.
    * `d_model`: [64, 128] – wymiar reprezentacji wewnętrznej w warstwach uwagi.
    * `d_ff`: [128, 256] – rozmiar warstwy *feed-forward*.
    * `dropout`: [0.0, 0.2] – współczynnik regularyzacji zapobiegający przeuczeniu.
    * `lr`: [10^-4, 10^-3] – współczynnik uczenia w skali logarytmicznej.

* **DLinear** (16 prób / trials)
    * `seq_len`: [48, 96, 192, 336, 512] – długość okna historycznego poddanego dekompozycji.
    * `individual`: [True, False] – flaga decydująca o stosowaniu unikalnych wag liniowych dla każdego kanału szeregu.
    * `lr`: [10^-4, 10^-2] – dopuszczalny szeroki zakres ze względu na wysoką stabilność modelu.

* **PatchTST** (10 prób / trials)
    * `context_length`: [48, 96, 192, 336, 512] – długość kontekstu wejściowego.
    * `num_hidden_layers`: [2, 3] – liczba bloków transformera.
    * `d_model`: [64, 128] – szerokość warstw ukrytych.
    * `head_dropout`: [0.0, 0.2] – regularyzacja końcowej głowicy prognozującej.
    * `learning_rate`: [10^-4, 10^-3].

* **iTransformer** (10 prób / trials)
    * `input_size`: [48, 96, 192, 336] – rozmiar wejściowy okna czasowego.
    * `hidden_size`: [64, 128] – rozmiar rzutowania zmiennych na tokeny.
    * `d_layers`: [1, 2] – liczba warstw enkodera.
    * `learning_rate`: [10^-4, 10^-3].

* **Autoformer** (6 prób / trials)
    * `input_size`: [48, 96] – drastycznie ograniczone ze względu na złożoność pamięciową.
    * `hidden_size`: [16, 32] – miniaturyzacja warstw w celu uniknięcia błędów OOM.
    * `dropout`: [0.0, 0.2].
    * `learning_rate`: [10^-4, 10^-3].

* **Temporal Fusion Transformer (TFT)** (6 prób / trials)
    * `input_size`: [48, 96] – limitowane ze względu na złożoną strukturę bramek GRN.
    * `hidden_size`: [16, 32] – rozmiar wektorów stanów ukrytych.
    * `dropout`: [0.1, 0.3] – wyższy zakres ze względu na skłonność TFT do przeuczenia na krótkich seriach.
    * `learning_rate`: [10^-4, 10^-3].

* **Chronos (T5-small)** (3 próby / trials)
    * `temperature`: [0.1, 0.8] – parametr kontrolujący „pewność” rozkładu tokenów.
    * `top_k`: [20, 80] – liczba najwyżej ocenianych próbek branych pod uwagę.
    * `top_p`: [0.8, 0.95] – parametryzacja próbkowania jądrowego (*nucleus sampling*).

* **AutoARIMA** (3 próby / trials)
    * `seasonal`: [True, False] – flaga włączająca automatyczne poszukiwanie komponentów sezonowych.

---

### Analiza procesu optymalizacji i ograniczeń sprzętowych

W badaniu zaobserwowano, że proces strojenia modeli prognozowania długookresowego jest silnie determinowany przez architekturę i jej interakcję z dostępną pamięcią wideo (VRAM). Modele o niskiej złożoności, jak **AutoARIMA** i **DLinear**, cechowały się zachowaniem deterministycznym i wysoką przewidywalnością. Dzięki relatywnie gładkiej funkcji kosztu, algorytm Optuna był w stanie zidentyfikować optymalne parametry w zaledwie kilkunastu próbach. W przypadku modelu **DLinear** kluczowym elementem był dobór parametru `individual`. Ustalono, że dla wielowymiarowych szeregów o unikalnej, niezależnej dynamice każdego kanału, separacja wag liniowych była krytyczna dla poprawy metryk, podczas gdy w zbiorach silnie skorelowanych wagi współdzielone działały jak naturalny filtr szumów, stabilizując prognozę.



Przejście do modeli opartych na mechanizmie uwagi (**Attention**) wprowadziło nową warstwę komplikacji: balans horyzontu kontekstowego. W modelach **CATS** i **PatchTST** sukces zależał od znalezienia „złotego środka” między rozmiarem okna historycznego a długością patcha. Model **CATS** wyróżniał się specyficznym procesem uczenia, który dzięki rezygnacji z self-attention na rzecz cross-attention oraz technice patchingu drastycznie redukuje liczbę tokenów wejściowych. Pozwoliło to na znacznie efektywniejsze trenowanie transformera przy zachowaniu wysokiej precyzji, co było kluczowe przy limicie **4 GB VRAM**. Z kolei model **iTransformer** okazał się przełomowy w danych wielowymiarowych, takich jak zbiór **Medical-M**. Dzięki traktowaniu zmiennych jako tokenów, model ten efektywniej wykorzystywał mniejszą liczbę warstw ukrytych, co skracało czas zbieżności i redukowało ryzyko przeuczenia, co widać w jego dominacji nad PatchTST w tej konkretnej domenie.

Największe wyzwania inżynieryjne wygenerowały modele o kwadratowej złożoności obliczeniowej $O(L^2)$, takie jak **Temporal Fusion Transformer (TFT)** oraz **Autoformer**. Architektury te na standardowym sprzęcie konsumenckim napotkały barierę technologiczną, wymuszającą redukcję rozmiaru paczki danych (`batch_size`) do wartości minimalnych (np. 2). Wprowadziło to do procesu uczenia silny szum stochastyczny, co zmuszało do stosowania bardzo niskich współczynników uczenia (rzędu $10^{-4}$). W efekcie, uzyskanie konkurencyjnych wyników dla tych modeli wymagało znacznie większej liczby prób w porównaniu do modeli liniowych, a każda sesja była obarczona wysokim ryzykiem błędu pamięci, co ostatecznie przełożyło się na ich gorszą wydajność w testach z długim horyzontem prognozy.



---

### Podsumowanie procesu tuningu

Przeprowadzone eksperymenty dowodzą, że inwestycja w maksymalną złożoność modelu rzadko zwraca się w warunkach ograniczonych zasobów obliczeniowych, ponieważ nakład pracy inżynieryjnej potrzebny na stabilizację „ciężkich” architektur często przewyższa realne zyski w metrykach. Modele klasyczne i liniowe, takie jak **AutoARIMA** i **DLinear**, oferują najlepszy stosunek jakości do nakładu pracy, stanowiąc bezkonkurencyjny punkt odniesienia, który w szeregach jednowymiarowych często okazuje się rozwiązaniem docelowym. W przypadku braku czasu na głęboki tuning, rozwiązaniem optymalnym okazuje się podejście „zero-shot” oferowane przez model **Chronos**, który zapewnia solidną bazę prognozy bez konieczności uczenia wag od podstaw.

Z kolei zaawansowane transformery o wysokiej złożoności, takie jak **Autoformer** czy **TFT**, wymagają do realnego pokazania swojej przewagi infrastruktury o znacznie większej pojemności (przykładowo jednostek klasy 16-32 GB VRAM). Przy dostępnych 4 GB ich potencjał jest fizycznie dławiony przez konieczność miniaturyzacji paczek danych i skracania okien wejściowych. Najbardziej uzasadnionym wyborem w klasie zaawansowanej pozostają zoptymalizowane modele **PatchTST**, **CATS** oraz **iTransformer**, które reprezentują „złoty środek” współczesnego prognozowania. Dzięki procesowi uczenia opartego na patchingu lub inwersji wymiarów, oferują one precyzję transformerów przy zachowaniu stabilności i niskiego kosztu pamięci. W środowisku o limitowanych zasobach sprzętowych znacznie większą wartość przynosi precyzyjny dobór okna kontekstowego w tych zoptymalizowanych modelach niż próby wymuszania stabilności na architekturach pokroju TFT, których proces optymalizacji w takich warunkach jest nieprzewidywalny i mało efektywny.

---

## 5. Wyniki eksperymentów

### Environment-M, Horizon=84

| Model                        | MAE (Total) | MSE (Total) | MAPE (Total) | MASE (Total) |
|------------------------------|-------------|-------------|--------------|--------------|
| CATS                         | 5.4889      | 38.2002     | 4.8%         | 3.1436       |
| DLinear                      | 2.7034      | 11.5950     | 2.4%         | 1.5483       |
| PatchTST                     | 5.3763      | 42.8862     | 4.7%         | 3.0792       |
| iTransformer                 | 1.9101      | 6.3731      | 1.7%         | 1.0940       |
| Autoformer                   | 2.0192      | 6.2956      | 1.8%         | 1.1565       |
| TemporalFusionTransformer    | 1.8816      | 6.9603      | 1.7%         | 1.0776       |
| AutoARIMA                    | 1.8936      | 6.2585      | 1.7%         | 1.0846       |
| Chronos (T5-small)           | 1.8342      | 5.9911      | 1.6%         | 1.0505       |

### Environment-M (N=333), Horizon=67

| Model                      | MAE (AVG) | MSE (AVG) | MAPE (AVG) | MASE (AVG) | MAE (TOTAL) | MSE (TOTAL) | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|-----------|-----------|------------|------------|------------|------------|-------------|--------------|
| CATS                      | 1.1911    | 4.9585    | 1.44e+15%  | 2.1993     | 2.8944     | 13.2168    | 2.6%        | 1.6904       |
| DLinear                   | 2.1302    | 13.8015   | 4.78e+15%  | 4.1673     | 6.1891     | 52.2887    | 5.4%        | 3.6145       |
| PatchTST                  | 0.8573    | 3.1103    | 7.20e+14%  | 1.2370     | 2.2497     | 8.8730     | 2.0%        | 1.3139       |
| iTransformer              | 1.0563    | 3.3315    | 2.48e+15%  | 2.9244     | 2.2608     | 8.0378     | 2.0%        | 1.3203       |
| Autoformer                | 0.9468    | 3.2172    | 9.95e+15%  | 2.3506     | 2.2776     | 9.1296     | 2.0%        | 1.3302       |
| TemporalFusionTransformer | 0.8954    | 2.6588    | 3.45e+14%  | 1.7409     | 2.3604     | 8.1493     | 2.0%        | 1.3785       |
| AutoARIMA                 | 0.9291    | 2.5917    | 19.5%      | 2.1012     | 2.4317     | 8.1903     | 2.1%        | 1.4201       |
| Chronos                   | 1.0256    | 3.0231    | 19.6%      | 2.3409     | 2.7401     | 10.2283    | 2.4%        | 1.6003       |


### Facility-M, Horizon 160

| Model                     | MAE (AVG)    | MSE (AVG)            | MAPE (AVG) | MASE (AVG) | MAE (TOTAL)  | MSE (TOTAL)            | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|--------------|----------------------|------------|------------|--------------|------------------------|--------------|--------------|
| CATS                      | 20350.0184   | 5406391385.6099      | 2.4%       | 0.4236     | 142041.0044  | 37834601865.5005       | 2.8%         | 0.6112       |
| DLinear                   | 23328.1407   | 6011624053.1917      | 4.3%       | 1.1654     | 162361.5287  | 42232986804.5243       | 3.2%         | 0.6987       |
| PatchTST                  | 38090.2758   | 10709931296.2281     | 4.1%       | 0.8703     | 265912.2829  | 74975730504.9744       | 5.2%         | 1.1443       |
| iTransformer              | 28572.9831   | 8271679806.4841      | 3.2%       | 0.7758     | 199476.4744  | 57895691046.3818       | 3.9%         | 0.8584       |
| Autoformer                | 39167.0773   | 11957656420.6679     | 4.3%       | 1.0992     | 273440.0048  | 83722351954.1749       | 5.3%         | 1.1767       |
| TemporalFusionTransformer | 33112.8919   | 15430712891.3305     | 3.9%       | 0.7260     | 231079.6035  | 107990699045.1924      | 4.3%         | 0.9944       |
| AutoARIMA                 | 37906.5131   | 10893631712.5329     | 3.8%       | 0.7360     | 264685.6413  | 76263287077.7721       | 5.2%         | 1.1390       |
| Chronos                   | 36686.6453   | 21473681932.7424     | 2.7%       | 0.5625     | 256462.7023  | 150421922683.0251      | 4.9%         | 1.1036       |


### Facility-M, Horizon 180

| Model                     | MAE (AVG)    | MSE (AVG)            | MAPE (AVG) | MASE (AVG) | MAE (TOTAL)  | MSE (TOTAL)            | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|--------------|----------------------|------------|------------|--------------|------------------------|--------------|--------------|
| CATS                      | 22976.9255   | 6139639133.9033      | 2.4%       | 0.4301     | 160435.2428  | 42972688650.0432       | 3.2%         | 0.6768       |
| DLinear                   | 20755.7843   | 5103541418.2197      | 3.7%       | 0.7388     | 144474.1174  | 35762161762.9978       | 2.9%         | 0.6095       |
| PatchTST                  | 19270.7475   | 4170871330.4841      | 3.7%       | 0.6543     | 134116.8838  | 29192997380.4260       | 2.7%         | 0.5658       |
| iTransformer              | 23508.6859   | 6143177607.6703      | 2.9%       | 0.7096     | 164073.8198  | 43000580126.8075       | 3.2%         | 0.6922       |
| Autoformer                | 40564.7820   | 11992972088.3679     | 3.9%       | 0.7509     | 283224.5353  | 83957002796.7066       | 5.6%         | 1.1948       |
| TemporalFusionTransformer | 29569.1807   | 17562418573.7251     | 3.7%       | 0.6880     | 206370.7856  | 122964785878.2375      | 3.8%         | 0.8706       |
| AutoARIMA                 | 33961.0939   | 9386275415.5679      | 3.7%       | 0.7025     | 237057.9802  | 65707021628.6972       | 4.6%         | 1.0001       |
| Chronos                   | 45618.8706   | 26652956541.8490     | 2.8%       | 0.5896     | 318890.1765  | 186672618320.5031      | 6.2%         | 1.3453       |


### Facility-M (N=650), Horizon=130

| Model                      | MAE (AVG) | MSE (AVG) | MAPE (AVG) | MASE (AVG) | MAE (TOTAL) | MSE (TOTAL) | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|-----------|-----------|------------|------------|------------|------------|-------------|--------------|
| CATS                      | 16534.156 | 3.58e+09  | 2.6%       | 0.5531     | 115316.599 | 2.51e+10   | 2.3%        | 0.4661       |
| DLinear                   | 22404.998 | 6.16e+09  | 3.7%       | 0.8065     | 156030.172 | 4.33e+10   | 3.1%        | 0.6307       |
| PatchTST                  | 17438.644 | 4.51e+09  | 3.6%       | 0.6273     | 121319.251 | 3.16e+10   | 2.4%        | 0.4904       |
| iTransformer              | 21102.034 | 5.77e+09  | 2.9%       | 0.6966     | 147217.757 | 4.04e+10   | 2.9%        | 0.5951       |
| Autoformer                | 39352.253 | 1.19e+10  | 4.0%       | 0.9233     | 274740.216 | 8.36e+10   | 5.4%        | 1.1105       |
| TemporalFusionTransformer | 30266.726 | 1.80e+10  | 3.8%       | 0.6755     | 211238.810 | 1.26e+11   | 3.9%        | 0.8538       |
| AutoARIMA                 | 37369.388 | 1.07e+10  | 3.7%       | 0.6362     | 260860.915 | 7.48e+10   | 5.1%        | 1.0544       |
| Chronos                   | 33338.257 | 1.91e+10  | 2.6%       | 0.4899     | 232821.857 | 1.34e+11   | 4.6%        | 0.9411       |


### Facility-U, Horizon 300

| Model                      | MAE | MSE  | MAPE                    | MASE |
|---------------------------|-----|------|--------------------------|------|
| CATS                      | 0.0448 | 0.0032 | 382908369294706.6% | 0.7712 |
| DLinear                   | 0.0468 | 0.0035 | 307091705167902.6% | 0.8043 |
| PatchTST                  | 0.0456 | 0.0032 | 237351323653492.9% | 0.7846 |
| iTransformer              | 0.0428 | 0.0035 | 300430613544985.5% | 0.7370 |
| Autoformer                | 0.0504 | 0.0039 | 356237382953673.5% | 0.8675 |
| TemporalFusionTransformer | 0.0497 | 0.0058 | 301600947393900.6% | 0.8551 |
| AutoARIMA                 | 0.0534 | 0.0036 | 409333022989200.5% | 0.9184 |
| Chronos                   | 0.0501 | 0.0059 | 298852235433324.7% | 0.8611 |


### Facility-U (N=1100), Horizon=220

| Model                      | MAE    | MSE     | MAPE      | MASE   |
|---------------------------|--------|---------|-----------|--------|
| CATS                      | 0.0515 | 0.0036  | 2.51e+14% | 0.8834 |
| DLinear                   | 0.0496 | 0.0035  | 2.77e+14% | 0.8519 |
| PatchTST                  | 0.0518 | 0.0035  | 2.31e+14% | 0.8886 |
| iTransformer              | 0.0502 | 0.0042  | 2.03e+14% | 0.8612 |
| Autoformer                | 0.0565 | 0.0047  | 2.64e+14% | 0.9705 |
| TemporalFusionTransformer | 0.0563 | 0.0063  | 2.07e+14% | 0.9666 |
| AutoARIMA                 | 0.0551 | 0.0038  | 2.77e+14% | 0.9457 |
| Chronos                   | 0.0566 | 0.0065  | 2.04e+14% | 0.9714 |



### Finance-U, Horizon=167

| Model                      | MAE    | MSE     | MAPE | MASE   |
|---------------------------|--------|---------|------|--------|
| CATS                      | 3.7176 | 22.7488 | 3.5% | 1.1347 |
| DLinear                   | 4.2633 | 28.6224 | 4.0% | 1.3013 |
| PatchTST                  | 3.4727 | 21.2530 | 3.2% | 1.0600 |
| iTransformer              | 3.9118 | 26.7721 | 3.6% | 1.1940 |
| Autoformer                | 3.4826 | 20.8469 | 3.2% | 1.0630 |
| TemporalFusionTransformer | 4.0073 | 31.0992 | 3.7% | 1.2231 |
| AutoARIMA                 | 3.4947 | 20.4898 | 3.3% | 1.0667 |
| Chronos                   | 4.6858 | 40.1994 | 4.3% | 1.4302 |


### Finance-U (N=667), Horizon=134

| Model                      | MAE    | MSE     | MAPE | MASE   |
|---------------------------|--------|---------|------|--------|
| CATS                      | 4.8893 | 40.3384 | 4.5% | 1.3460 |
| DLinear                   | 3.8772 | 23.4035 | 3.6% | 1.0674 |
| PatchTST                  | 3.5888 | 19.6374 | 3.3% | 0.9880 |
| iTransformer              | 4.1063 | 29.5107 | 3.8% | 1.1305 |
| Autoformer                | 3.5064 | 20.7925 | 3.3% | 0.9653 |
| TemporalFusionTransformer | 3.6302 | 25.7311 | 3.3% | 0.9994 |
| AutoARIMA                 | 3.4395 | 19.6748 | 3.2% | 0.9469 |
| Chronos                   | 5.1688 | 44.7193 | 4.8% | 1.4230 |


### Medical-M, Horizon=160

| Model                      | MAE (AVG) | MSE (AVG) | MAPE (AVG) | MASE (AVG) | MAE (TOTAL) | MSE (TOTAL) | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|-----------|-----------|------------|------------|------------|------------|-------------|--------------|
| CATS                      | 0.2499    | 0.1531    | 31.5%      | 1.5179     | 0.4380     | 0.3566     | 58.3%       | 1.4165       |
| DLinear                   | 0.2504    | 0.1419    | 30.5%      | 1.7418     | 0.4769     | 0.3973     | 59.2%       | 1.5425       |
| PatchTST                  | 0.2039    | 0.1049    | 29.7%      | 1.4782     | 0.3452     | 0.2572     | 56.7%       | 1.1165       |
| iTransformer              | 0.2052    | 0.1075    | 26.1%      | 1.4699     | 0.3900     | 0.2937     | 50.5%       | 1.2615       |
| Autoformer                | 0.2123    | 0.1097    | 27.3%      | 1.5204     | 0.4025     | 0.3001     | 53.5%       | 1.3017       |
| TemporalFusionTransformer | 0.1889    | 0.1024    | 25.2%      | 1.4304     | 0.3550     | 0.2693     | 51.6%       | 1.1480       |
| AutoARIMA                 | 0.2082    | 0.1068    | 27.1%      | 1.3942     | 0.3603     | 0.2724     | 51.5%       | 1.1654       |
| Chronos                   | 0.2418    | 0.2029    | 29.4%      | 1.5319     | 0.4401     | 0.4664     | 59.6%       | 1.4234       |


### Medical-M, Horizon=180

| Model                      | MAE (AVG) | MSE (AVG) | MAPE (AVG) | MASE (AVG) | MAE (TOTAL) | MSE (TOTAL) | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|-----------|-----------|------------|------------|------------|------------|-------------|--------------|
| CATS                      | 0.2451    | 0.1482    | 31.5%      | 1.7380     | 0.4514     | 0.3867     | 57.1%       | 1.4490       |
| DLinear                   | 0.2296    | 0.1280    | 29.1%      | 1.4666     | 0.4185     | 0.3310     | 51.3%       | 1.3434       |
| PatchTST                  | 0.2084    | 0.1003    | 28.3%      | 1.5481     | 0.3447     | 0.2363     | 48.0%       | 1.1065       |
| iTransformer              | 0.1941    | 0.0996    | 25.3%      | 1.2701     | 0.3514     | 0.2549     | 48.6%       | 1.1281       |
| Autoformer                | 0.2131    | 0.1098    | 28.0%      | 1.5951     | 0.3965     | 0.2922     | 49.2%       | 1.2728       |
| TemporalFusionTransformer | 0.2110    | 0.1111    | 28.5%      | 1.7520     | 0.3937     | 0.2954     | 49.8%       | 1.2636       |
| AutoARIMA                 | 0.2377    | 0.1192    | 29.7%      | 1.7241     | 0.4598     | 0.3426     | 51.8%       | 1.4758       |
| Chronos                   | 0.2150    | 0.1644    | 28.3%      | 1.4043     | 0.3822     | 0.3575     | 54.8%       | 1.2269       |


### Medical-M (N=650), Horizon=130

| Model                      | MAE (AVG) | MSE (AVG) | MAPE (AVG) | MASE (AVG) | MAE (TOTAL) | MSE (TOTAL) | MAPE (TOTAL) | MASE (TOTAL) |
|---------------------------|-----------|-----------|------------|------------|------------|------------|-------------|--------------|
| CATS                      | 0.2741    | 0.1384    | 33.9%      | 2.3287     | 0.4947     | 0.3888     | 53.7%       | 1.5757       |
| DLinear                   | 0.3149    | 0.2139    | 37.0%      | 2.0522     | 0.5911     | 0.5622     | 56.8%       | 1.8829       |
| PatchTST                  | 0.2215    | 0.1085    | 26.9%      | 1.6470     | 0.4050     | 0.2886     | 53.5%       | 1.2900       |
| iTransformer              | 0.1925    | 0.0932    | 23.7%      | 1.5025     | 0.3458     | 0.2386     | 49.6%       | 1.1014       |
| Autoformer                | 0.2045    | 0.0986    | 24.9%      | 1.5415     | 0.3737     | 0.2701     | 50.1%       | 1.1905       |
| TemporalFusionTransformer | 0.2179    | 0.1051    | 26.6%      | 1.8070     | 0.4093     | 0.2950     | 50.6%       | 1.3036       |
| AutoARIMA                 | 0.2561    | 0.1225    | 30.2%      | 2.0509     | 0.4946     | 0.3707     | 52.9%       | 1.5753       |
| Chronos                   | 0.2416    | 0.1800    | 30.1%      | 1.8339     | 0.4190     | 0.4362     | 52.3%       | 1.3346       |


### WebService-U, Horizon=152

| Model                      | MAE    | MSE      | MAPE  | MASE   |
|---------------------------|--------|----------|-------|--------|
| CATS                      | 3.6743 | 20.6483  | 13.2% | 2.2158 |
| DLinear                   | 9.2980 | 102.5519 | 32.2% | 5.6070 |
| PatchTST                  | 4.6726 | 31.6022  | 15.5% | 2.8177 |
| iTransformer              | 7.2081 | 72.9567  | 23.6% | 4.3468 |
| Autoformer                | 9.2917 | 130.5044 | 29.4% | 5.6032 |
| TemporalFusionTransformer | 8.7848 | 119.0615 | 27.6% | 5.2976 |
| AutoARIMA                 | 11.7793| 188.8596 | 38.0% | 7.1034 |
| Chronos                   | 9.4042 | 128.7082 | 29.8% | 5.6711 |


### WebService-U (N=608), Horizon=122

| Model                      | MAE     | MSE      | MAPE | MASE   |
|---------------------------|---------|----------|------|--------|
| CATS                      | 11.5761 | 169.8835 | 37.5%| 6.8239 |
| DLinear                   | 8.8727  | 94.3758  | 30.1%| 5.2303 |
| PatchTST                  | 3.0094  | 14.0752  | 10.1%| 1.7740 |
| iTransformer              | 5.6562  | 52.9939  | 18.0%| 3.3342 |
| Autoformer                | 9.3992  | 130.7419 | 28.8%| 5.5407 |
| TemporalFusionTransformer | 10.2142 | 149.8309 | 31.4%| 6.0211 |
| AutoARIMA                 | 7.1367  | 73.9912  | 29.9%| 4.2070 |
| Chronos                   | 4.5630  | 30.4379  | 16.3%| 2.6898 |


### 5.2 Attention vs brak attention
- Porównanie:
  - DLinear vs PatchTST
  - PatchTST vs CATS
- Wpływ usunięcia self-attention na jakość prognozy

## 6. Dyskusja wyników
- Czy attention poprawia jakość prognoz?
- Czy proste modele liniowe są konkurencyjne?
- Jak zachowują się modele dla długiego horyzontu?

## 7. Wnioski
- Najważniejsze obserwacje z eksperymentów
- Odpowiedź na pytanie z tematu zadania
- Krótkie podsumowanie

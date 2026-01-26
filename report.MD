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

## 4. Wyniki eksperymentów

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


### 4.2 Attention vs brak attention
- Porównanie:
  - DLinear vs PatchTST
  - PatchTST vs CATS
- Wpływ usunięcia self-attention na jakość prognozy

## 5. Dyskusja wyników
- Czy attention poprawia jakość prognoz?
- Czy proste modele liniowe są konkurencyjne?
- Jak zachowują się modele dla długiego horyzontu?

## 6. Wnioski
- Najważniejsze obserwacje z eksperymentów
- Odpowiedź na pytanie z tematu zadania
- Krótkie podsumowanie

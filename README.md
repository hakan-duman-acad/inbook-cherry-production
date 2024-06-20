# Application of Machine Learning Algorithms for Univariate Time Series Analysis in Agricultural Forecasting: A Case Study of Cherry Production in Turkey

## Abstract

**Keywords:**

## R Packages

This study used the R statistical environment, version 4.2.2, developed
by R Core Team ([2022](#ref-r-2022)). The tidyverse meta-package,
version 2.0.0, created Wickham et al. ([2019](#ref-tidyverse-2019)), was
employed for data manipulation and cleaning. For time series data
extension, the tsibble package (version 1.1.3), developed by Wang, Cook,
& Hyndman ([2020](#ref-tsibble-2020)), was utilized. To build
forecasting models, the fable package (version 0.3.3) created by
O’Hara-Wild, Hyndman, & Wang ([2023a](#ref-fable-2023)) was employed.
For feature extraction and statistical analysis, the feasts package
(version 0.3.1), developed by O’Hara-Wild, Hyndman, & Wang
([2023b](#ref-feasts-2023)), was utilized. To create world maps,
rnaturalearth version 0.3.4 by Massicotte & South
([2023](#ref-rnaturalearth-2023)), rnaturalearthdata version 0.1.0 by
South ([2017](#ref-rnaturalearthdata-2017)),sf package version 1.0.14
and sp package version 2.1.2 with contributions from Pebesma & Bivand
([2005](#ref-rnews-2005)) and Bivand, Pebesma, & Gomez-Rubio
([2013](#ref-asdar-2013)), Pebesma ([2018](#ref-sp-2018)) were employed.

## Acknowledgements

This analysis adapted and modified code from various sources, such as
books, package manuals, vignettes, and GitHub repositories. The sources
are cited as follows:

-   Data preparing, manipulation, cleaning, and visualization: Wickham
    et al. ([2019](#ref-tidyverse-2019)), Wang et al.
    ([2020](#ref-tsibble-2020)), Wang & contibutors
    ([2024](#ref-tsibble-2024-github)),
-   Map Visualization: Massicotte & South
    ([2023](#ref-rnaturalearth-2023)), South
    ([2017](#ref-rnaturalearthdata-2017)), Pebesma & Bivand
    ([2005](#ref-rnews-2005)), Bivand et al. ([2013](#ref-asdar-2013)),
    Pebesma & contibutors ([2024](#ref-sf-2024-github))
-   Training and Forecasting Models: Kuhn & Silge
    ([2022](#ref-kuhn-2022)), Kuhn & Wickham
    ([2020](#ref-tidymodels-2020)), Kuhn & Wickham
    ([2024](#ref-tidymodels-2024-github)), Dancho & Vaughan
    ([2024](#ref-timetk-2024-github)), Dancho
    ([2024](#ref-modeltime-2024-github)), Hyndman, Koehler, Ord, &
    Snyder ([2008](#ref-hyndman-2008)), Eyduran, Ertürk, Duman, & Aliyev
    ([2020](#ref-univariate-R-2020))

## Code References

Bivand, R. S., Pebesma, E. J., & Gomez-Rubio, V. (2013). *Applied
spatial data analysis with R, second edition*. Springer, NY.
<https://asdar-book.org/>

Dancho, M. (2024). *Modeltime: The tidymodels extension for time series
modeling*. <https://github.com/business-science/modeltime> R package
version 1.3.0, https://business-science.github.io/modeltime/

Dancho, M., & Vaughan, D. (2024). *Timetk: A tool kit for working with
time series*. <https://github.com/business-science/timetk> R package
version 2.9.0, https://business-science.github.io/timetk/

Eyduran, E., Ertürk, Y. E., Duman, H., & Aliyev, P. (2020). Examples of
univariate time series analysis with artificial neural networks in r.
<https://doi.org/10.13140/RG.2.2.36747.31528/1>

Hyndman, R. J., Koehler, A., Ord, K., & Snyder, R. (2008). *Forecasting
with exponential smoothing: The state space approach* (p. 359). Berlin:
Springer.

Kuhn, M., & Silge, J. (2022). *Tidy modeling with r: A framework for
modeling in the tidyverse*. " O’Reilly Media, Inc.".
<https://www.tmwr.org/>

Kuhn, M., & Wickham, H. (2020). *Tidymodels: A collection of packages
for modeling and machine learning using tidyverse principles.*
<https://www.tidymodels.org>

Kuhn, M., & Wickham, H. (2024). *Tidymodels-org*.
<https://github.com/tidymodels> R package version 1.1.1,
https://www.tidymodels.org/

Massicotte, P., & South, A. (2023). *Rnaturalearth: World map data from
natural earth*. <https://CRAN.R-project.org/package=rnaturalearth> R
package version 0.3.4

O’Hara-Wild, M., Hyndman, R., & Wang, E. (2023a). *Fable: Forecasting
models for tidy time series*. <https://CRAN.R-project.org/package=fable>
R package version 0.3.3

O’Hara-Wild, M., Hyndman, R., & Wang, E. (2023b). *Feasts: Feature
extraction and statistics for time series*.
<https://CRAN.R-project.org/package=feasts> R package version 0.3.1

Pebesma, E. J. (2018). <span class="nocase">Simple Features for R:
Standardized Support for Spatial Vector Data</span>. *The R Journal*,
*10*(1), 439–446. <https://doi.org/10.32614/RJ-2018-009>

Pebesma, E. J., & Bivand, R. (2005). Classes and methods for spatial
data in R. *R News*, *5*(2), 9–13.
<https://CRAN.R-project.org/doc/Rnews/>

Pebesma, E. J., & contibutors. (2024). Simple features for R. June 2,
2025, <https://r-spatial.github.io/sf/>

R Core Team. (2022). *R: A language and environment for statistical
computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>

South, A. (2017). *Rnaturalearthdata: World vector map data from natural
earth used in ’rnaturalearth’*.
<https://CRAN.R-project.org/package=rnaturalearthdata> R package version
0.1.0

Wang, E., & contibutors. (2024). Tidyverts/tsibble. June 2, 2025,
<https://github.com/tidyverts/tsibble>

Wang, E., Cook, D., & Hyndman, R. J. (2020). A new tidy data structure
to support exploration and modeling of temporal data. *Journal of
Computational and Graphical Statistics*, *29*(3), 466–478.
<https://doi.org/10.1080/10618600.2019.1695624>

Wickham, H., Averick, M., Bryan, J., Chang, W., McGowan, L. D.,
François, R., … Yutani, H. (2019). Welcome to the <span
class="nocase">tidyverse</span>. *Journal of Open Source Software*,
*4*(43), 1686. <https://doi.org/10.21105/joss.01686>

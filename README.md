ShortRate_Model
==================

Implement a one-factor model called [Kalotay-Williams-Fabozzi](https://en.wikipedia.org/wiki/Short-rate_model#One-factor_short-rate_models), discretizing it using a [Binomial Tree](https://en.wikipedia.org/wiki/Lattice_model_%28finance%29#Interest_rate_derivatives). The model uses a single stochastic factor – the short rate – to determine the future evolution of all interest rates. This assignment was done as part of the Master in [Quantitative Finance ](http://eesp.fgv.br/ensino/mestrado-profissional/economia/area-financas-quantitativas) course, at the [FGV](https://en.wikipedia.org/wiki/Fundação_Getúlio_Vargas) University. You can check our report <a href="https://nbviewer.jupyter.org/github/ucaiado/ShortRate_Model/blob/master/notebooks/Kalotay-Williams-Fabozzi.ipynb" target="_blank">here</a> (the text is in Portuguese). We will use mostly Python to code the project.


### Install
This project requires **Python 2.7** and the following Python libraries installed:

- [Matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Scipy](https://www.scipy.org/)
- [Seaborn](https://web.stanford.edu/~mwaskom/software/seaborn/)


### Reference
1. A. J. Kalotay, G. O. Williams, F. J. Fabozzi.  *A Model For Value Bonds and Embedded Options*.   Financial Analystis Journal, 1993. [*link*](http://www.kalotay.com/sites/default/files/private/FAJ93.pdf)
2. G. W. Buetow Jr., B. Hanke, F. J. Fabozzi.  *Impact of Different Interest Rate Models on Bond Value Measures*. The Journal Of Fixed Income, 2001. [*link*](http://www.iijournals.com/doi/abs/10.3905/jfi.2001.319304?journalCode=jfi)

### License
The contents of this repository are covered under the [MIT License](LICENSE).

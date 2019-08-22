all: exercises-01-python.pdf exercises-02-python-cont.pdf exercises-03-classes.pdf exercises-04-numpy.pdf junk

include = include.tex

%.pdf: %.tex
	xelatex $<
	xelatex $<


junk:
	rm *.aux *.log *.dvi *.ps -f

clean:
	rm *.aux *.log *.dvi *.ps -f
	rm exercises-*.pdf -f


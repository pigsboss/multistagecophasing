# 由 primer-book 生成；改完 .tex 可直接 make 重编。
JOB  = lunar_swing_summary
ROOT = ../..
OUT  = _primer/book
LATEX = xelatex
PDF  = xdvipdfmx

all : note

note :
	cd $(ROOT) && $(LATEX) -interaction=nonstopmode -no-pdf -output-directory=$(OUT) $(OUT)/$(JOB)
	cd $(ROOT) && $(LATEX) -interaction=nonstopmode -no-pdf -output-directory=$(OUT) $(OUT)/$(JOB)
	cd $(ROOT) && $(LATEX) -interaction=nonstopmode -no-pdf -output-directory=$(OUT) $(OUT)/$(JOB)
	cd $(ROOT) && $(PDF) -o $(OUT)/$(JOB).pdf $(OUT)/$(JOB).xdv

clean :
	rm -f $(OUT)/$(JOB).log $(OUT)/$(JOB).aux $(OUT)/$(JOB).toc
	rm -f $(OUT)/$(JOB).lof $(OUT)/$(JOB).lot $(OUT)/$(JOB).out $(OUT)/$(JOB).xdv

distclean : clean
	rm -f $(OUT)/$(JOB).pdf

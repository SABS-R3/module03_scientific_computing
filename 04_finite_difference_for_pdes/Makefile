
lectures_dir = lectures
practicals_dir = exercises
publish_dir = publish
solution_dir = $(practicals_dir)/solution
solution_files = $(solution_dir)/*.py ${solution_dir}/requirements.txt

all: ${publish_dir}/exercise.pdf  ${publish_dir}/solution.zip

${publish_dir}/solution.zip: $(solution_files)
	zip $@ $(solution_files) 

reveal_command = "pandoc -f markdown$(extensions) -t revealjs -s ${reveal_options} $^ -o ${publish_dir}/$@.html"
syntax =--highlight-style pygments_custom.theme

${publish_dir}/lecture_%.pdf:${lectures_dir}/%.md
	pandoc -f markdown$(extensions) -t beamer ${syntax} $^ -V theme:metropolis -V aspectratio:169 -o $@

project%:${projects_dir}/project%.md
	pandoc  -f markdown$(extensions) $(pdf_template) $^ -o ${publish_dir}/$@.pdf

${publish_dir}/exercise.pdf: ${practicals_dir}/exercises.tex
	cd ${practicals_dir}; xelatex exercises.tex
	cp ${practicals_dir}/exercises.pdf ${publish_dir}/

clean:
	rm ${publish_dir}/*.pdf

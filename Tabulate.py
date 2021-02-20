from string import Template

# This is an example of how to use template strings to produce a latex
# table.
#
# Katharine Long, TTU, Nov 2020.
#
# To use this from your own code, do
# 'from Tabulate import LatexSafeTemplate'.


# Because we need to use a non-standard placeholder delimiter ('??' instead of '$')
# we need to make a subclass of Template that overrides the default delimiter
# attribute. This class does just that.
#
# You, the user, will just use LatexSafeTemplate instead of Template.
#
# If you want to understand how this works, read up on subclasses and class variables.
# Otherwise, you can simply trust that it works and use it.
class LatexSafeTemplate(Template):

    delimiter = '??'

    def __init__(self, templateStr):
        Template.__init__(self, templateStr)




# ------------ Example -----------------------------------------------------

if __name__=='__main__':

    # Example template string for a latex code fragment.
    #
    # I do *not* recommend
    # putting your whole latex file into a template string; instead, just write
    # a fragment of latex code for the table format you're going to reuse for
    # different runs.
    #
    # In this example, I've used '??' for the placeholder delimiter instead of the
    # default '$' because you might use '$' in your latex code.
    #
    # IMPORTANT: Both python and latex use the backslash ('\') character in special
    # ways, so you have to deal with it carefully. In a python string, the substring
    # '\b' is used for the backspace symbol; if you want it to mean, literally, "\b"
    # (as will be the case in, for example, \begin{table} in latex) you need to
    # escape the backslash: that is, write '\\begin' instead of '\begin'. If you
    # need to write "\\" in latex (as in ending lines), you will need
    # to write '\\\\' in the python
    # template because '\\' is escaped to a single backslash in a python string.
    # The backslashes to watch out for are:
    # '\', '\'', '\n', '\r', '\t', '\f', '\o', and '\x'
    # all of which are parsed by python as escapes for special characters. So if
    # you intend to write the latex code '\frac{1}{2}' you need to escape
    # the backslash using a double-backslash, '\\frac{1}{2}' or else python
    # with interpret '\f' as a formfeed character.

    # With those preliminary comments, here is a template for a simple 2 by 3
    # latex table. The placeholders are called a, b, c, d, e, f. Note the
    # double backslashes in \\begin and in \\\\, for the reasons explained above.
    myString = """
    \\begin{tabular}{|c|c|c|}
    \hline
    ??a & ??b & ??c \\\\
    \hline
    ??d & ??e & ??f \\\\
    \hline
    \end{tabular}
    """




    # Create a template object.
    t = LatexSafeTemplate(myString)

    # Now that we have a template object, we need to assign values to the placeholders.

    # We need to make a dictionary that maps placeholder names ('a', 'b', ...)
    # to values. You can do this by hand for each table; however, since I'm lazy,
    # I'll write a single function to fill in values of the two rows in the table.

    def varDict(row1, row2):
        return {
        'a' : row1[0],
        'b' : row1[1],
        'c' : row1[2],
        'd' : row2[0],
        'e' : row2[1],
        'f' : row2[2]
        }

    # Let's make our first table. Here are row values.
    tab1Row1 = [1, 2, 3]
    tab1Row2 = [2.3, 4.5, 6.7]

    # Do the substitutions and write to a file 'table1.tex'. Your main latex file
    # should do \input{table1.tex} in the appropriate place to include the table
    # you just produced.
    with open('table1.tex', 'w') as f:
        table1 = t.substitute(varDict(tab1Row1, tab1Row2))
        f.write(table1)

    # Let's make another table. The values don't have to be numbers.
    tab2Row1 = ['Einstein', 'Newton', 'Pythagoras']
    tab2Row2 = ['$E=mc^2$', '$F=ma$', '$a^2+b^2=c^2$']

    # Do the substitutions and write to a file 'table2.tex'
    with open('table2.tex', 'w') as f:
        table2 = t.substitute(varDict(tab2Row1, tab2Row2))
        f.write(table2)


    # All done. If you run pdflatex from the command line, you can uncomment
    # the following and this script will do all the latex'ing of 'main.tex' and build a PDF
    # for you. If you process your latex in some other way, then simply
    # do whatever you normally do, or, if you want to script that, replace pdflatex
    # by whatever command you run.

    #import os
    #os.system('pdflatex ./main.tex')

# -*- coding: utf-8 -*-
"""
===========
mathematicamagic
===========

Magics for interacting with Mathematica via pythonica.

.. note::

  The ``pythonica`` and ``mathlink`` module needs to be installed separately and
  can be obtained on github and packaged with the Mathematica software
  respecitively. 

Usage
=====

``%mathematica``

{MATHEMATICA_DOC}

``%mathematica_push``

{MATHEMATICA_PUSH_DOC}

``%mathematica_pull``

{MATHEMATICA_PULL_DOC}

"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2012 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#  Basically all of pythonica.py so this runs correctly in one file
#-----------------------------------------------------------------------------

import mathlink as _ml
import time as _time

__author__="""\n""".join(['Benjamin Edwards (bedwards@cs.unm.edu)'])

#    Copyright (C) 2012
#    Benjamin Edwards
#    All rights reserved.
#    BSD license

# Packets that signify incoming tokens
_incoming_token = [_ml.RETURNPKT,
                   _ml.RETURNEXPRPKT,
                   _ml.DISPLAYPKT,
                   _ml.DISPLAYENDPKT,
                   _ml.RESUMEPKT,
                   _ml.RETURNTEXTPKT,
                   _ml.SUSPENDPKT,
                   _ml.MESSAGEPKT]

#identity function for anything to strings
_id_to_mathematica = lambda x: str(x)

#Convert a float to a string for mathematica
def _float_to_mathematica(x):
    return ("%e"%x).replace('e','*10^')

#Convert a complex to a string for mathematica
def _complex_to_mathematica(z):
    return 'Complex' + ('[%e,%e]'%(z.real,z.imag)).replace('e','*10^')

#convert some type of container to a string for matheatica
def _iter_to_mathematica(xs):
    s = 'List['
    for x in xs:
        s += _python_mathematica[type(x)](x)
        s += ','
    s = s[:-1]
    s+= ']'
    return s

#Convert a string to a mathematica string
def _str_to_mathematica(s):
    return '\"%s\"'%s

#Dictionary for type conversions.
_python_mathematica = {bool:_id_to_mathematica,
                       type(None):_id_to_mathematica,
                       int:_id_to_mathematica,
                       float:_float_to_mathematica,
                       long:_id_to_mathematica,
                       complex:_complex_to_mathematica,
                       iter:_iter_to_mathematica,
                       list:_iter_to_mathematica,
                       set:_iter_to_mathematica,
                       xrange:_iter_to_mathematica,
                       str:_str_to_mathematica,
                       tuple:_iter_to_mathematica,
                       frozenset:_iter_to_mathematica}

#Take a string from mathematica and try to make it into a python object
#This could likely be written better and in the future could include
#methods for other functional conversions
def _mathematica_str_python(s):
    if s == 'Null' or s is None:
        return None
    try:
        val = int(s)
    except ValueError:
        try:
            val = float(s)
        except ValueError:
            try:
                val = float(s.replace('*10^','e'))
            except ValueError:
                val = None
    # Some sort of Number, so return it NEED TO ADD COMPLEX and Rational
    if val is not None:
        return val
    val = {}
    s = s.replace(" ","").replace('{','List[').replace('}',']')
    open_brack = s.find("[")
    #Some String not a function Call, likely rational,complex,list or symbol
    if open_brack == -1:
        div = s.find('/')
        if div != -1:
            try:
                num = _mathematica_str_python(s[:div])
                den = _mathematica_str_python(s[div+1:])
                if num/den == float(num)/den:
                    return num/den
                else:
                    return float(num)/den
            except TypeError:
                val = s
        im = s.find('I')
        if im == -1:
            val = s
        else:
            plus = s.find('+')
            times = s.find('*I')
            if plus != -1:
                if times != -1:
                    try:
                        return complex(_mathematica_str_python(s[:plus]),
                                       _mathematica_str_python(s[plus+1:times]))
                    except TypeError:
                        val = s
                else:
                    try:
                        return complex(_mathematica_str_python(s[:plus]),1)
                    except TypeError:
                        val = s
            else:
                if times != -1:
                    try:
                        return complex(0,_mathematica_str_python(s[:times]))
                    except TypeError:
                        val = s
                else:
                    return complex(0,1)
        return val
    func = s[:open_brack]
    num_open_brack = 1
    val[func] = [] 
    last_comma = open_brack
    for i in range(open_brack+1,len(s)):
        if s[i] == ',' and num_open_brack == 1:
            val[func].append(_mathematica_str_python(s[last_comma+1:i]))
            last_comma = i
        elif s[i] == '[':
            num_open_brack += 1
        elif s[i] == ']':
            if num_open_brack > 1:
                num_open_brack -= 1
            elif num_open_brack == 1:
                val[func].append(_mathematica_str_python(s[last_comma+1:len(s)-1]))
            else:
                raise Exception("Unbalanced Brackets")
    if func == 'List':
        return val['List']
    elif func == 'Complex':
        return complex(val['Complex'][0],val['Complex'][1])
    elif func == 'Rational':
        return float(val['Rational'][0])/val['Rational'][1]
    else:
        return val

#Searches Mathematica string of type 'InputForm' for things to plot
def _find_plot_strings(s):
    ps = []
    for g_func in ['Graphics[','Graphics3D[','Image[','Grid[']:
        while True:
            graph_start = s.find(g_func)
            if graph_start == -1:
                break
            num_brack = 1
            for i in range(graph_start+len(g_func),len(s)):
                if s[i] == '[':
                    num_brack += 1
                elif s[i] == ']':
                    if num_brack == 1:
                        ps.append(s[graph_start:i+1])
                        break
                    else:
                        num_brack -= 1
            s = s.replace(s[graph_start:i+1],'')
    return ps


#Exception
class PythonicaException(Exception):
    pass

class Pythonica(object):
    """
    Base class for Mathematica Communication.

    Creates a link to a Mathematica Kernel and stores information needed
    communication

    Parameters
    ----------
    name : string
      String to launch mathlink.
    mode : string
      Sting for mode to launch mathlink
    timeout : int
      Time to give Mathematica to start the kernel
    debug : Bool
      Whether to print debug information
    plot_dir : string
      Directory to store plots
    plot_size : tuple of 2 ints
      Tuple containing plot size in pixels. If None let's Mathematica decide
      what size to make things
    plot_format : string
      Format for plots, default to 'png'. 'bmp', 'svg', and 'jpeg' tested and
      seem to work.
    output_prompt : string
      Whether to print output prompts reported from Mathematica
    input_prompt : string
      Whether to print input prompts reported from Mathematica

    Examples
    --------
    >>> import pythonica
    >>> m = pythonica.Pythonica()
    >>> m.eval('Mean[{1,2,3}]')
    '2'
    """


    def __init__(self,
                 name='math -mathlink',
                 mode='launch',
                 timeout=1,
                 debug=False,
                 plot_dir=None,
                 plot_size=None,
                 plot_format='png',
                 output_prompt=False,
                 input_prompt=False):
        self._env = _ml.env()
        self.kernel = self._env.open(name,mode=mode)
        self.kernel.connect()
        self.debug=debug
        self.plot_dir = plot_dir
        self.plot_num = 0
        self.last_python_result=None
        self.last_str_result=None
        self.plot_size = plot_size
        self.plot_format = plot_format
        self.output_prompt = output_prompt
        self.input_prompt = input_prompt
        self.last_error = None
        _time.sleep(timeout)
        if not self.kernel.ready():
            raise PythonicaException("Unable to Start Mathematica Kernel")
        else:
            packet = self.kernel.nextpacket()
            if self.debug:
                print _ml.packetdescriptiondictionary[packet]
            if packet == _ml.INPUTNAMEPKT:
                self.kernel.getstring()

    def eval(self,expression,make_plots=True,output_type='string',str_format='input'):
        """
        Evaluate a string in the Mathematica Kernel

        Evalutes the string 'expression' in Mathematica.

        Parameters
        ----------
        expression: string
          Expression to be evaluated
        make_plots: boolean
          Whether to produce plots, plot_dir must not be None
        output_type: string
          Whether to output a string or a python object, must be either
          'string' or 'pythong'
        str_format: string
          How to format the string if output_type='string'. If 'input' will
          produce a string which can be used as Mathematica Input. If 'tex'
          will produce valid tex. If 'plain' will produce whatever plain text
          mathematica would produce.

        Returns
        -------
        String or python object.

        Raises
        ------
        PythonicaException

        Examples
        --------

        >>> m.eval('D[Log[x],x]')
        'x^(-1)'
        >>> m.eval('Mean[{1,2,3,4}]',output_type='python')
        2.5
        >>> m.eval('D[Log[x],x]',str_output='tex')
        '\\\\frac{1}{x}'
        >>> print m.eval('D[Log[x],x]',str_output='plain')
        1
        -
        x

        See Also
        --------
        README.rst
        """
        self.last_python_result=None
        self.last_str_result=None
        self.last_error=None
        if str_format=='tex':
            expression = 'ToString[' + expression+',TeXForm]'
        elif str_format=='input':
            expression = 'ToString[' + expression + ',InputForm]'
        elif str_format=='plain':
            pass
        else:
            raise PythonicaException("String Format must be 'tex', 'input', or 'plain'")
        self.kernel.putfunction("EnterTextPacket",1)
        self.kernel.putstring(expression)
        self.__parse_packet()
        str_result = self.last_str_result
        if self.last_error is not None:
            raise PythonicaException(self.last_error.decode('string_escape'))
        if make_plots and self.plot_dir is not None:
            plot_exp = _find_plot_strings(str_result)
            for s in plot_exp:
                filename='\"%s/pythonica_plot_%i.%s\"'%(self.plot_dir,self.plot_num,self.plot_format)
                if self.plot_size is None:
                    self.eval('Export[%s,%s];'%(filename,s),make_plots=False,str_format='plain')
                else:
                    (w,h) = self.plot_size
                    self.eval('Export[%s,%s,ImageSize->{%i,%i}];'%(filename,s,w,h),make_plots=False,str_format='plain')
                self.plot_num += 1
        if str_format == 'plain' and str_result is not None:
            str_result = str_result.decode('string_escape')
        self.last_str_result = str_result
        if output_type == 'python':
            self.last_python_result = _mathematica_str_python(str_result)
            return self.last_python_result
        elif output_type == 'string':
            self.last_python_result = None
            if str_result == 'Null':
                return None
            else:
                return str_result
        else:
            raise PythonicaException("Output Type must be either 'python' or 'string'(default)")

    def push(self, name, value):
        """
        Push python object to Mathematica Kernel.

        Can make some conversions of python objects to Mathematica. See
        README.rst for more information.

        Parameters
        ----------
        name : string
          Name for value in Mathematica Kernel
        value : python object
          Object to be pushed to Mathematica Kernel

        Returns
        -------
        None

        Raises
        ------
        PythonicaException: If the object cannot be converted

        Examples
        --------

        >>> m.push('l',[1,2,3])
        >>> m.eval('l2 = 2*l;')
        >>> m.pull('l2')
        [2,4,6]
        """

        convert_function = _python_mathematica.get(type(value),-1)
        if convert_function is -1:
            raise PythonicaException("Could not convert %s to Mathematica Object"%type(value))
        s = 'Set[%s,%s];'%(name,convert_function(value))
        self.eval(s,make_plots=False)

    def pull(self,name):
        """
        Return a Mathematica Object to the python environment.

        Parameters
        ---------
        name: string
          Name to retrieve

        Returns
        -------
        python object:
          Depending on type will be converted. See README.rst for more info

        Examples
        --------

        >>> m.eval('X = List[1,2,3,4]')
        >>> m.pull('X')
        [1,2,3,4]
        """
        res = self.eval(name,make_plots=False)
        return _mathematica_str_python(res)

    def __parse_packet(self):
        if self.debug:
            print("in __parse_packet")
        packet = self.kernel.nextpacket()
        if self.debug:
            print _ml.packetdescriptiondictionary[packet]
        if packet == _ml.INPUTNAMEPKT:
            if self.input_prompt:            
                print(self.kernel.getstring())
            else:
                self.kernel.getstring()
            return None 
        elif packet == _ml.OUTPUTNAMEPKT:
            if self.output_prompt:
                print(self.kernel.getstring())
            else:
                self.kernel.getstring()
            self.__parse_packet()
        elif packet == _ml.MESSAGEPKT:
            if self.last_error is None:
                self.last_error = self.kernel.getstring()
            else:
                self.last_error += "\t" + self.kernel.getstring()
            self.__parse_token(packet)
            self.__parse_packet()
        elif packet == _ml.TEXTPKT:
            self.last_error += self.kernel.getstring()
            self.__parse_packet()
        elif packet == _ml.SYNTAXPKT:
            self.kernel.getstring()
            self.__parse_packet()
        elif packet in _incoming_token:
            if self.debug:
                print("Going to Parse Token")
            self.last_str_result = self.__parse_token(packet).replace(r'\\\012','').replace(r'\012>   ','')
            self.__parse_packet()
        else:
            raise PythonicaException("Unknown Packet %s"%_ml.packetdescriptiondictionary[packet])


    def __parse_token(self,packet):
        if self.debug:
            print("In Parse Token")
        try:
            token = self.kernel.getnext()
            if self.debug:
                print _ml.tokendictionary[token]
        except _ml.error, e:
            raise PythonicaException("Got Error Token: %s"%e)
        if token == _ml.MLTKSTR:
            return self.kernel.getstring()
        else:
            raise PythonicaException("Unknown Token %i",token)

    def __del__(self):
        self.kernel.close()


import tempfile
from glob import glob
from shutil import rmtree
from xml.dom import minidom

from IPython.core.displaypub import publish_display_data
from IPython.core.magic import (Magics, magics_class, line_magic,
                                line_cell_magic)
from IPython.testing.skipdoctest import skip_doctest
from IPython.core.magic_arguments import (
    argument, magic_arguments, parse_argstring
)
from IPython.utils.py3compat import unicode_to_str

_mimetypes = {'png' : 'image/png',
              'svg' : 'image/svg+xml',
              'jpeg': 'image/jpeg',
              'jpg' : 'images/jpeg',
              'bmp' : 'image/bmp'}

@magics_class
class MathematicaMagics(Magics):
    """A set of magics useful for interactive work with Mathematica via
    pythonica. 
    """
    def __init__(self, shell):
        """
        Parameters
        ----------
        shell : IPython shell

        """
        super(MathematicaMagics, self).__init__(shell)
        self._mathematica = Pythonica()
        # Allow publish_display_data to be overridden for
        # testing purposes.
        self._publish_display_data = publish_display_data

    def _fix_gnuplot_svg_size(self, image, size=None):
        """
        Mathematica SVGs do not have height/width attributes in the correct
        place. Set as the actual plot size, which is sometimes hidden among the
        xml

        Parameters
        ----------
        image : str
            SVG data.
        size : tuple of int
            Image width, height.

        """
        (svg,) = minidom.parseString(image).getElementsByTagName('svg')
        try:
            (rect,) = minidom.parseString(image).getElementsByTagName('image')
        except:
            rect = minidom.parseString(image).getElementsByTagName('rect')[1]

        w = rect.getAttribute('width')
        h = rect.getAttribute('height')

        if size is not None:
            width, height = size 
        else:
            width, height = int(w),int(h)

        svg.setAttribute('width', '%dpx' % width)
        svg.setAttribute('height', '%dpx' % height)
        return svg.toxml()

    @skip_doctest
    @line_magic
    def mathematica_push(self, line):
        '''
        Line-level magic that pushes a variable to Mathematica.

        `line` should be made up of whitespace separated variable names in the
        IPython namespace::

            In [7]: import numpy as np

            In [8]: X = range(5)

            In [9]: X.mean()
            Out[9]: 2.0

            In [10]: %mathematica_push X

            In [11]: %mathematica Mean[X]
            Out[11]: 2.0

        '''
        inputs = line.split(' ')
        for input in inputs:
            input = unicode_to_str(input)
            self._mathematica.push(input, self.shell.user_ns[input])


    @skip_doctest
    @line_magic
    def mathematica_pull(self, line):
        '''
        Line-level magic that pulls a variable from Mathematica.

            In [18]: _ = %mathematica x = {{1,2}, {3,4}}; y = 'hello';

            In [19]: %mathematica_pull x y

            In [20]: x
            Out[20]:
            array([[ 1.,  2.],
                   [ 3.,  4.]])

            In [21]: y
            Out[21]: 'hello'

        '''
        outputs = line.split(' ')
        for output in outputs:
            output = unicode_to_str(output)
            self.shell.push({output: self._mathematica.pull(output)})


    @skip_doctest
    @magic_arguments()
    @argument(
        '-i', '--input', action='append',
        help='Names of input variables to be pushed to Mathematica. Multiple names '
             'can be passed, separated by commas with no whitespace.'
        )
    @argument(
        '-o', '--output', action='append',
        help='Names of variables to be pulled from Mathematica after executing cell '
             'body. Multiple names can be passed, separated by commas with no '
             'whitespace.'
        )

    @argument(
        '-f', '--format', action='store',
        help='Format for Graphic outputs')

    @argument(
        '-s', '--size', action='store',
        help='Pixel size of plots, "width,height". Default is None, which '
        'allows Mathematica to determine the size'
        )
    @argument(
        '-p', '--print_output',action='store',
        help='Whether to print the output from Mathematica Execution')

    @argument(
        '-t', '--str_format',action='store',
        help='Whether to output a string which can be used as input(string)'+\
             'tex(tex) or mathematica formatted(plain)')

    @argument(
        'code',
        nargs='*',
        )

    @line_cell_magic
    def mathematica(self, line, cell=None):
        '''
        Execute code in Mathematica, and pull some of the results back into the
        Python namespace.

            In [9]: %mathematica X = {{1,2}, {3,4}}; Mean[X]
            Out[9]: {2, 3}

        As a cell, this will run a block of Mathematica code.::

            In [10]: %%mathematica
                   : fx = Log[x]
               ....: D[fx,x]

            Power[x,-1]

        In the notebook, plots are published as the output of the cell, e.g.

        %mathematica Plot[Sin[x],{x,0,10}]

        will create a line plot.

        Objects can be passed back and forth between Mathematica and IPython via the
        -i and -o flags in line::

            In [14]: Z = [1, 4, 5, 10]

            In [15]: %mathematica -i Z Mean[Z]
            Out[15]: array([ 5.])


            In [16]: %mathematica -o W W = Z * Mean[Z]
            Out[16]: [  5,  20,  25,  50]

            In [17]: W
            Out[17]: [  5,  20,  25,  50]

        The size of output plots can be specified::

            In [18]: %%mathematica -s 600,800
                ...: ListPlot[{1, 2, 3},Joined->True];

        '''
        args = parse_argstring(self.mathematica, line)
        # arguments 'code' in line are prepended to the cell lines
        if cell is None:
            code = ''
            return_output = True
        else:
            code = cell
            return_output = False

        code = ' '.join(args.code) + code

        if args.input:
            for input in ','.join(args.input).split(','):
                input = unicode_to_str(input)
                self._mathematica.push(input,self.shell.user_ns[input])

        # generate plots in a temporary directory
        if args.size is not None:
            size = tuple(map(int,args.size.split(',')))
            self._mathematica.plot_size=size
        else:
            size = None 
            self._mathematica.plot_size=None

        plot_dir = tempfile.mkdtemp()
        self._mathematica.plot_dir = plot_dir

        if args.format is not None:
            self._mathematica.plot_format=args.format
        else:
            self._mathematica.plot_format='png'

        if args.print_output is not None:
            print_output = (True,False)[args.print_output=="False"]  
        else:
            print_output = True

        if args.str_format is not None:
            str_format = args.str_format
        else:
            str_format = 'input'
        
        plot_format=self._mathematica.plot_format
        result = self._mathematica.eval(code,str_format=str_format)
        if str_format == 'tex' or str_format == 'plain':
            output = None
        else:
            output = _mathematica_str_python(result)
        key = 'MathematicaMagic.Mathematica'
        display_data = []

        # Publish images
        images = [open(imgfile, 'rb').read() for imgfile in \
                  glob("%s/*.%s"%(plot_dir,plot_format))]
        rmtree(plot_dir)

        plot_mime_type = _mimetypes.get(plot_format, 'image/png')
        for image in images:
            if self._mathematica.plot_format == 'svg':
                image = self._fix_gnuplot_svg_size(image,size=size)
            display_data.append((key, {plot_mime_type: image}))

        if args.output:
            for output in ','.join(args.output).split(','):
                output = unicode_to_str(output)
                self.shell.push({output: self._mathematica.pull(output)})

        for source, data in display_data:
            self._publish_display_data(source, data)

        if str_format=='tex':
            tex = '$' + result + '$'
            self._publish_display_data(key,{'text/latex':tex.decode('string_escape')})
            return_output = False

        if return_output and print_output:
            if output is None:
                return result
            else:
                return output

__doc__ = __doc__.format(
    MATHEMATICA_DOC = ' '*8 + MathematicaMagics.mathematica.__doc__,
    MATHEMATICA_PUSH_DOC = ' '*8 + MathematicaMagics.mathematica_push.__doc__,
    MATHEMATICA_PULL_DOC = ' '*8 + MathematicaMagics.mathematica_pull.__doc__
    )


_loaded = False
def load_ipython_extension(ip):
    """Load the extension in IPython."""
    global _loaded
    if not _loaded:
        ip.register_magics(MathematicaMagics)
        _loaded = True

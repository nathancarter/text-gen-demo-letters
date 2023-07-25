
import streamlit as st
import pandas as pd

##############
#
#  Global Constants
#

max_gen_len = 100
max_tuple_len = 5
start_token = ' '
end_token = '.'
words_we_care_about = '''
apple ask bicycle bread crank close deal different eel else frank fool
great go hello holes ignore if joke jolly kelp koala llama love mush me
never night owl out prod place quack quite ring roast say stop trick two
under up vacation violin wet wild xylophone xantham you yes zebra zoo
'''.replace( '\n', ' ' ).strip().split( ' ' )

# generate training data
sentences = [
	f'{word[0].upper()+word[1:]} begins with the letter {word[0].upper()}'
	for word in words_we_care_about
]


##############
#
#  Utility Functions
#

# function that fetches tuples from the training data, of arbitrary size
def tuple_table ( num_prev_words=1 ):
	rows = [ ]
	for sentence in sentences:
		tokens = [ start_token ] * num_prev_words \
			   + sentence.split( ' ' ) \
			   + [ end_token ]
		for index in range( num_prev_words, len( tokens ) ):
			rows.append( tokens[index-num_prev_words:index+1] )
	colnames = [ 'Next', 'Previous' ]
	for count in range( 2, num_prev_words+1 ):
		colnames.append( f'{count} words ago' )
	colnames.reverse()
	return pd.DataFrame( rows, columns = colnames )

# create a frequency table from the set of tuples in the training data
def tall_frequency_table ( num_prev_words=1 ):
	base = tuple_table( num_prev_words )
	base['Frequency'] = 1
	base = base.groupby( base.columns[:-1].tolist() )[['Frequency']].agg( 'count' )
	return base

# in the case where num_prev_words==1, we can represent the table in wide form, too
def wide_frequency_table ():
	result = base_frequency_table().pivot_table(
		index='Previous', columns='Next',
		values='Frequency', aggfunc='sum' ).fillna( 0 )
	if len( result ) == 0:
		return result
	# move START row to the top
	result = pd.concat( [
		result.loc[[start_token],:],
		result.drop(start_token, axis=0)
	], axis=0 )
	# move END row to the left
	columns = [ col for col in result.columns if col is not end_token ]
	result = result[[ end_token, *columns ]]
	# clean up and be done
	result.index.name = None
	return result

# generate a random sentence from the given table of tuples
def generate_sentence ( tuple_table ):
	n = len( tuple_table.columns ) - 1
	sentence = [ start_token ] * n
	while sentence[-1] is not end_token and len( sentence ) < max_gen_len:
		subset = df
		for i in range( 0, n ):
			subset = subset[subset.iloc[:,i].isin( [ sentence[-n+i] ] )]
		sentence.append( subset['Next'].sample( 1 ).iloc[0] )
	sentence = ' '.join( sentence[n:-1] )
	return sentence[0].upper() + sentence[1:] + '.'


##############
#
#  User Interface
#

# title
st.title( 'Very Small Word Hallucinator' )

# structure of the app as a set of tabs
text_tab, matrix_tab, generator_tab = st.tabs( [
	'View training data',
	'Learned frequencies',
	'Text generation'
] )

# show tab that lets users see the current training data
with text_tab:
	if len( sentences ) > 0:
		st.write( '\n'.join( [
		    ' 1. ' + sentence[0].upper() + sentence[1:] + '.'
			for sentence in sentences
		] ) )
	else:
		st.write( '(No training data entered yet.)' )

# show tab that lets users see the frequency tables computed from the training data
with matrix_tab:
	if len( sentences ) == 0:
		st.warning( 'No training data available yet.  Add some first.' )
	else:
		n = st.number_input( 'Number of words to use for prediction',
							 1, max_tuple_len, 1, key='matrix_tab_n' )
		st.write( '(Scroll down to see full table.)' )
		df = tall_frequency_table( n )
		df['Frequency'] = df['Frequency'].astype( float ) # for Streamlit Cloud
		st.dataframe(
			df,
			use_container_width=True,
			column_config={
				'Frequency' : st.column_config.ProgressColumn(
					max_value=df['Frequency'].max(), format='%d' )
			}
		)
		# st.write( wide_frequency_table() )

# show tab that generates random new sentences based on what was learned
with generator_tab:
	if len( sentences ) == 0:
		st.warning( 'No training data available yet.  Add some first.' )
	else:
		n = st.number_input( 'Number of words to use for prediction',
							 1, max_tuple_len, 1, key='generator_tab_n' )
		if st.button( 'Generate some sentences' ):
			df = tuple_table( n )
			st.success( '\n'.join( [
				f' 1. {generate_sentence(df)}' for _ in range(10)
			] ) )


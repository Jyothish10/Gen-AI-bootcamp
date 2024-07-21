import streamlit as st
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.llms.bedrock import Bedrock

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    #amazon.titan-text-lite-v1 model
    llm=Bedrock(
        credentials_profile_name='default',#use your aws creditinals

        model_id="amazon.titan-text-lite-v1",
        model_kwargs={
            "maxTokenCount": 200,
            "temperature": 0.7,
            "topP": 0.9
        }
    )
    
    ## Prompt Template

    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response






st.set_page_config(page_title="Generate Blogs",
                    
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Hey,Generate content for your blogs")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People','Student'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
#Packages.
library(shiny)
library(shinythemes)
library(zipcode)
library(dplyr)


df_doctors <- read.csv(file="Doctors4.csv", header=TRUE, sep=",")
Model <- read.csv(file="model.csv", header=TRUE, sep=",")


shinyUI(navbarPage("Skin.ly", theme = shinytheme("cerulean"),

                   
                   
                   #Upload picture tab
                   tabPanel("Upload your picture",
                            h1("Picture"),
                            "Please Upload your picture to analyse the type of cancer.",
                            hr(),
                            
                            
                            fileInput('file1', 'Choose file to upload',
                                      accept = c('jpg')
                            ),
                            
                            fluidRow(column(6, verbatimTextOutput("picture"))),
                            fluidRow(column(6, dataTableOutput("modelresult")))
                            
                            
                   )
                   ,
                   
#Read data

#Patient tab
tabPanel("Patient information",

h1("Patient information"),
"Welcome to Heathcare System App enter your information to get recomendation on specialist in your area.",
hr(),


fluidRow(column(8,
                # Copy the line below to make a text input box
                textInput("Name", label = h4("Name"), value = "Enter your name..."))),

fluidRow(column(8,
                # Copy the line below to make a text input box
                numericInput("Age", label = h4("Age"), 0))),

fluidRow(column(8,
                # Copy the line below to make a text input box
                selectInput("state",h4("State:"), df_doctors$State_Code_of_the_Provider))),

fluidRow(column(8,
           #Copy the line below to make a text input box
          numericInput("Zip", label = h4("ZipCode"), 7090))),



hr(),

h1("Doctor Recomendation"),


("Please indicate your zip code to search specialist near your area"),

" ",           

fluidRow(column(8, dataTableOutput("Zip"))),

         hr()
      
      )
,
    


#Abou tab
tabPanel("About Us",
         h1("About Us"),
         "Hackaton submition",
         hr()
         
       )

,



#bottom page
#    hr(),
#    hr(), 
#    hr(),
#    hr(),
    h6("2019-School of AI hackaton"),
    h6("Victor Ruiz"),
    h6("Jorge Guerra"),
    h6("Andreina Torres"),
    h6("Prasanth Babu")
    

 )

)
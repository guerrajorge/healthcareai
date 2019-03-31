##################################Library###########################
library(shiny)


##################################LoadData###########################
df_doctors <- read.csv(file="Doctors4.csv", header=TRUE, sep=",")
Model <- read.csv(file="model.csv", header=TRUE, sep=",")


#############################Funtions#########################################

library(zipcode)



list_extract <- function(column, position = 1, rownum = 1){
  
  if(position==0){
    stop("position 0 is not supported")
  }
  
  if(is.data.frame(column[[1]])){
    if(position<0){
      sapply(column, function(column){
        index <- ncol(column) + position + 1
        if(is.null(column[rownum, index]) | index <= 0){
          # column[rownum, position] still returns data frame if it's minus, so position < 0 should be caught here
          NA
        } else {
          column[rownum, index][[1]]
        }
      })
    } else {
      sapply(column, function(column){
        if(is.null(column[rownum, position])){
          NA
        } else {
          column[rownum, position][[1]]
        }
      })
    }
  } else {
    if(position<0){
      sapply(column, function(column){
        index <- length(column) + position + 1
        if(index <= 0){
          # column[rownum, position] still returns data frame if it's minus, so position < 0 should be caught here
          NA
        } else {
          column[index]
        }
      })
    } else {
      sapply(column, function(column){
        column[position]
      })
    }
  }
}

get_geo_distance = function(long1, lat1, long2, lat2, units = "miles") {
  loadNamespace("purrr")
  loadNamespace("geosphere")
  longlat1 = purrr::map2(long1, lat1, function(x,y) c(x,y))
  longlat2 = purrr::map2(long2, lat2, function(x,y) c(x,y))
  distance_list = purrr::map2(longlat1, longlat2, function(x,y) geosphere::distHaversine(x, y))
  distance_m = list_extract(distance_list, position = 1)
  if (units == "km") {
    distance = distance_m / 1000.0;
  }
  else if (units == "miles") {
    distance = distance_m / 1609.344
  }
  else {
    distance = distance_m
    # This will return in meter as same way as distHaversine function. 
  }
  distance
}





################################# Server logic ######################################


shinyServer(function(input, output) {
 
  
  Zip <- renderTable({ input$Zip })
  Name <- renderTable({ input$Name })
  Age <- renderTable({input$Age })
  #Zip<-input$Zip

  colnames(df_doctors) <-  c("Identifier",	"Type","Lastname",	"Name","Middle","Credentials","gender","Address","city","zip","state","country")
 
   df_doctorsZip <- reactive({ 
   df_doctors[df_doctors$state == input$state, c(1,2,3,4,8,10,11)] })
   latitude<-reactive({ zipcode[zipcode$zip==input$Zip,4]})
   longitude<- reactive({ zipcode[zipcode$zip==input$Zip,5]})
  #recomendation<-apply(df_doctorsZip, 1, get_geo_distance,long1=longitude, lat1=latitude,long2=df_doctorsZip$longitude, lat2=df_doctorsZip$latitude)

                                                           
  output$Zip<-renderDataTable({df_doctorsZip()})


  
  picture <-  renderText({input$file1$name})
  output$picture <-  renderText({input$file1$name})
  
  
  modelresult <- reactive({ 
    req(input$file1)    
    
  python_path="/anaconda3/envs/py3/bin/python"
  image_path=paste("/Users/ruizv1/Desktop/temp/temp/", input$file1$name, sep="")
  model_path="/Users/ruizv1/Desktop/temp/temp/model.hd5"
  script_path="/Users/ruizv1/GitRepos/challenge/predict_from_pic.py"
  out_path="/Users/ruizv1/Desktop/prediction.txt"
  
  call_s = paste(python_path," ", script_path, " --image_path=", image_path, " --model_path=", model_path, " --out_path=", out_path, sep="")
  
  system(call_s)
  
  as.character(read.csv(out_path, header=F)$V1[1])
  

  })
  output$modelresult<-renderDataTable({ data.frame(prediction=c(modelresult())) })
  
  
})





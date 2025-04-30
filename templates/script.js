// Name : Choi Siu Hin
// SID: 1155157707


let content = document.getElementById("content");
let button = document.getElementById("Special");
let task1 = document.getElementById("task1");
let task2 = document.getElementById("task2");
let task3 = document.getElementById("task3");
let a=1;
let switch1 = document.getElementById("switch1");
let switch2 = document.getElementById("switch2");
let switch3 = document.getElementById("switch3");
let switch4 = document.getElementById("switch4");
let switch5 = document.getElementById("switch5");
let pattern = /^[^ ]+@[^ ]+\.[a-z]{2,3}$/;
let newEmail123 = document.getElementById("new-email");
let newComment123 = document.getElementById("new-comment")
let form123=document.getElementById("form");
let b=1;
let progressBar=document.getElementById("progressBar");
let progressBlock=document.getElementById("progressBlock");
let stickyID= document.getElementById("stickyID");
progressBar.style.display="none";
progressBlock.style.display="none";
stickyID.style.display="none";
let emailValid=0;
let commentValid=0;

function update(){
    progressBar.style.width = `${((window.scrollY) / (document.body.scrollHeight - window.innerHeight) * 100)}%`;
    requestAnimationFrame(update);
}

update();


let setError = (newElement, message) => {
    let errorDisplay = newElement.parentElement.querySelector('.error');
    
    errorDisplay.innerText = message;
    newElement.classList.add("is-invalid");
}

let setSuccess = newElement => {
    let errorDisplay = newElement.parentElement.querySelector('.error');

    errorDisplay.innerText = '';
    newElement.classList.remove("is-invalid");
}; 



button.onclick = function(){
    if(content.className=="show"){
        content.className="";
        task1.style.display="none";
        task2.style.display="none";
        task3.style.display="none";
    }
    else{
         content.className="show";
         task1.style.display="block";
         task2.style.display="block";
         task3.style.display="block";
    } 
}


function task1Function(){
    if(a==1){
         a=2;
         switch1.className="textCenter";
         switch2.className="textCenter";
         switch3.className="textCenter";
         switch4.className="textCenter";
         switch5.className="textCenter";
    }
    else if (a==2){
        a=3;
         switch1.className="textRight";
         switch2.className="textRight";
         switch3.className="textRight";
         switch4.className="textRight";
         switch5.className="textRight";
    }
    else {
        a=1;
        switch1.className="textLeft";
        switch2.className="textLeft";
        switch3.className="textLeft";
        switch4.className="textLeft";
        switch5.className="textLeft";
    }
}

function task2Function(){
    let newHobby=prompt("What is the new hobby?");
    let inputHobby= document.createElement("div");
    let element0308='<div class="m-2 p-3 bg-primary"> ' +    ' </div>';
    inputHobby.innerHTML = element0308;
    inputHobby.querySelector("div").innerHTML = newHobby;
    document.getElementById("123hobby").appendChild(inputHobby);
}

function task3Function(){
    if(b==1){
        b=2;
        progressBar.style.display="block";
        progressBlock.style.display="block";
        stickyID.style.display="block";
    }
    else {
        b=1;
        progressBar.style.display="none";
        progressBlock.style.display="none";
        stickyID.style.display="none";
    }
}



function processform() {

    if(document.querySelector("#new-email").value === '') {
        setError(newEmail123, 'Email is required');
        emailValid=0;
    } else if (!document.querySelector("#new-email").validity.valid) {
        setError(newEmail123, 'Provide a valid email address');
        emailValid=0;
    } else {
        setSuccess(newEmail123);
        emailValid=1;
    }
    
    if(document.querySelector("#new-comment").value === '') {
        setError(newComment123, 'This cannot be empty');
        commentValid=0;
    } else {
        setSuccess(newComment123);
        commentValid=1;
    }
     
    if(emailValid==0 || commentValid==0){
        return;
    }
    let newComment= document.createElement("div");
    let element = '<div>  ' + 
        '                 <svg height="100" width= "100"> <circle cx="50" cy="50" r="40"> </svg>' +
        '          </div>' +
        '          <div>' +
        '               <h5> </h5>' +
        '               <p> </p>' +
        '          </div>';
        newComment.innerHTML = element;

        newComment.className= "d-flex";
        newComment.querySelectorAll("div")[0].className = "flex-shrink-0";
        newComment.querySelectorAll("div")[1].className = "flex-grow-1";

        let lastComment = document.querySelector("#comments").lastElementChild;
    
        newComment.querySelector("h5").innerHTML = document.querySelector("#new-email").value;
        newComment.querySelector("p").innerHTML = document.querySelector("#new-comment").value;
        
        let color = document.querySelectorAll("input[name=new-color]:checked")[0].value;

        newComment.querySelector("circle").setAttribute("fill", color);

        document.querySelector("#comments").appendChild(newComment);
        
        fetch('file.txt', {method: 'Put', body: document.querySelector("#new-email").value + '\n'+ document.querySelector("#new-comment").value+ '\n' + document.querySelectorAll("input[name=new-color]:checked")[0].value + '\n'});
        
        document.getElementById("form").reset();
};



function loadfile() {
    fetch('file.txt').then(res => res.text()).then(txt => { let storeTxt=txt; let storing = ["","","",""]; let j=0;
    for (var i = 0; i < storeTxt.length; i++) {
        if(storeTxt.charAt(i)=="\n"){
             j++;
        }
        else {
            storing[j]= storing[j] + storeTxt.charAt(i);
        }
      }
      console.log(storeTxt.length);
      console.log(storing[0]);
      console.log(storing[1]);
      console.log(storing[2]);
      console.log(txt);
      
      let newComment= document.createElement("div");
      let element = '<div>  ' + 
        '                 <svg height="100" width= "100"> <circle cx="50" cy="50" r="40"> </svg>' +
        '          </div>' +
        '          <div>' +
        '               <h5> </h5>' +
        '               <p> </p>' +
        '          </div>';
        newComment.innerHTML = element;
        newComment.className= "d-flex";
        newComment.querySelectorAll("div")[0].className = "flex-shrink-0";

        newComment.querySelectorAll("div")[1].className = "flex-grow-1";
    
        newComment.querySelector("h5").innerHTML = storing[0];
        newComment.querySelector("p").innerHTML = storing[1];
        
        let color = storing[2];

        newComment.querySelector("circle").setAttribute("fill", color);

        document.querySelector("#comments").appendChild(newComment); 
    })
};

loadfile();

        
        
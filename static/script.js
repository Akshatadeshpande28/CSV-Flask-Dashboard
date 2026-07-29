document.addEventListener("DOMContentLoaded", function () {

const btn=document.getElementById("themeToggle");

if(localStorage.getItem("theme")=="dark"){

document.body.classList.add("dark-mode");

btn.innerHTML='<i class="bi bi-sun-fill"></i> Light Mode';

}

btn.addEventListener("click",function(){

document.body.classList.toggle("dark-mode");

if(document.body.classList.contains("dark-mode")){

localStorage.setItem("theme","dark");

btn.innerHTML='<i class="bi bi-sun-fill"></i> Light Mode';

}else{

localStorage.setItem("theme","light");

btn.innerHTML='<i class="bi bi-moon-stars-fill"></i> Dark Mode';

}

});

});
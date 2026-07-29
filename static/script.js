document.addEventListener("DOMContentLoaded", function () {

    const themeToggle = document.getElementById("themeToggle");

    // Check previously saved theme
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "dark") {
        document.body.classList.add("dark-mode");

        if (themeToggle) {
            themeToggle.innerHTML =
                '<i class="bi bi-sun-fill me-1"></i> Light Mode';
        }
    }


    // Dark mode button
    if (themeToggle) {

        themeToggle.addEventListener("click", function () {

            document.body.classList.toggle("dark-mode");

            if (document.body.classList.contains("dark-mode")) {

                localStorage.setItem("theme", "dark");

                themeToggle.innerHTML =
                    '<i class="bi bi-sun-fill me-1"></i> Light Mode';

            } else {

                localStorage.setItem("theme", "light");

                themeToggle.innerHTML =
                    '<i class="bi bi-moon-stars-fill me-1"></i> Dark Mode';
            }

        });

    }

});
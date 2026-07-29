document.addEventListener(
    "DOMContentLoaded",
    function () {

        // ================================================
        // DARK MODE
        // ================================================

        const themeToggle =
            document.getElementById("themeToggle");


        const savedTheme =
            localStorage.getItem("theme");


        if (savedTheme === "dark") {

            document.body.classList.add(
                "dark-mode"
            );


            if (themeToggle) {

                themeToggle.innerHTML =
                    '<i class="bi bi-sun-fill me-1"></i> Light Mode';

            }

        }


        if (themeToggle) {

            themeToggle.addEventListener(
                "click",
                function () {

                    document.body.classList.toggle(
                        "dark-mode"
                    );


                    const dark =
                        document.body.classList.contains(
                            "dark-mode"
                        );


                    if (dark) {

                        localStorage.setItem(
                            "theme",
                            "dark"
                        );


                        themeToggle.innerHTML =
                            '<i class="bi bi-sun-fill me-1"></i> Light Mode';

                    } else {

                        localStorage.setItem(
                            "theme",
                            "light"
                        );


                        themeToggle.innerHTML =
                            '<i class="bi bi-moon-stars-fill me-1"></i> Dark Mode';

                    }

                }
            );

        }


        // ================================================
        // FILE SELECTION
        // ================================================

        const fileInput =
            document.getElementById("fileInput");

        const selectedFile =
            document.getElementById("selectedFile");


        if (fileInput && selectedFile) {

            fileInput.addEventListener(
                "change",
                function () {

                    if (
                        fileInput.files &&
                        fileInput.files.length > 0
                    ) {

                        const file =
                            fileInput.files[0];


                        selectedFile.innerHTML =
                            '<i class="bi bi-file-earmark-check-fill me-2"></i>'
                            + file.name;


                        selectedFile.classList.remove(
                            "d-none"
                        );

                    }

                }
            );

        }


        // ================================================
        // DRAG & DROP VISUAL EFFECT
        // ================================================

        const dropArea =
            document.getElementById("dropArea");


        if (dropArea && fileInput) {

            [
                "dragenter",
                "dragover"
            ].forEach(
                function (eventName) {

                    dropArea.addEventListener(
                        eventName,
                        function (event) {

                            event.preventDefault();

                            dropArea.classList.add(
                                "dragging"
                            );

                        }
                    );

                }
            );


            [
                "dragleave",
                "drop"
            ].forEach(
                function (eventName) {

                    dropArea.addEventListener(
                        eventName,
                        function (event) {

                            event.preventDefault();

                            dropArea.classList.remove(
                                "dragging"
                            );

                        }
                    );

                }
            );


            dropArea.addEventListener(
                "drop",
                function (event) {

                    const files =
                        event.dataTransfer.files;


                    if (
                        files &&
                        files.length > 0
                    ) {

                        fileInput.files = files;


                        if (selectedFile) {

                            selectedFile.innerHTML =
                                '<i class="bi bi-file-earmark-check-fill me-2"></i>'
                                + files[0].name;


                            selectedFile.classList.remove(
                                "d-none"
                            );

                        }

                    }

                }
            );

        }


        // ================================================
        // FILTER TYPE SWITCHING
        // ================================================

        const filterColumn =
            document.getElementById(
                "filterColumn"
            );


        const categoricalFilter =
            document.getElementById(
                "categoricalFilter"
            );


        const numericFilters =
            document.querySelectorAll(
                ".numeric-filter"
            );


        function updateFilterType() {

            if (!filterColumn) {
                return;
            }


            const selectedOption =
                filterColumn.options[
                    filterColumn.selectedIndex
                ];


            const isNumeric =
                selectedOption &&
                selectedOption.dataset.numeric
                === "true";


            if (isNumeric) {

                if (categoricalFilter) {

                    categoricalFilter.classList.add(
                        "d-none"
                    );

                }


                numericFilters.forEach(
                    function (element) {

                        element.classList.remove(
                            "d-none"
                        );

                    }
                );

            } else {

                if (categoricalFilter) {

                    categoricalFilter.classList.remove(
                        "d-none"
                    );

                }


                numericFilters.forEach(
                    function (element) {

                        element.classList.add(
                            "d-none"
                        );

                    }
                );

            }

        }


        if (filterColumn) {

            filterColumn.addEventListener(
                "change",
                updateFilterType
            );


            updateFilterType();

        }

    }
);
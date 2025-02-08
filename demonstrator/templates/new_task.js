const datasetLoaders = {{ dataset_loaders|tojson|safe }};
const baseTask = {{ base_task|tojson|safe }};

function updateDatasetParameters() {
    const loaderName = document.getElementById('dataset_loader').value;
    const parametersContainer = document.getElementById('dataset-parameters');
    parametersContainer.innerHTML = '';
    document.getElementById('dataset-description').innerHTML = datasetLoaders[loaderName].description;
    if(datasetLoaders[loaderName]) {
        const fields = datasetLoaders[loaderName].parameters;
        for(const param of fields) {
            const options = datasetLoaders[loaderName].parameter_options[param] || null;
            const inputGroup = createInputField(param[0], param[1], param[2], param[3], options);
            parametersContainer.appendChild(inputGroup);
        }
        if(baseTask && baseTask["dataset_loader"] === loaderName) for(const paramName in baseTask["dataset_parameters"]) {
            const input = document.getElementsByName(paramName)[0];
            if(input) input.value = baseTask.dataset_parameters[paramName];
        }
    }
    checkFormCompletion();
}

function checkFormCompletion() {
    let allFilled = true;
    document
        .getElementById('task-form-data')
        .querySelectorAll('input[required]')
        .forEach(input => {if (!input.value) allFilled = false;});
    if(document.getElementById('dataset_loader').value === "") allFilled = false;
    document.getElementById('next-button-data').disabled = !allFilled;
}

function createTitle(param, name, width='250px') {
    const title = document.createElement('span');
    title.innerHTML = name;
    title.htmlFor = param;
    title.style.marginRight = '10px';
    title.style.width = width;
    return title;
}

function createSelection(param, options) {
    const select = document.createElement('select');
    select.name = param;
    select.className = 'form-control';
    options.forEach(optionValue => {
        const option = document.createElement('option');
        option.value = optionValue;
        option.textContent = optionValue;
        if(optionValue == def) option.selected = true;
        select.appendChild(option);
    });
    return select;
}

function createInputField(param, type, def, title_text, options=null) {
    const div = document.createElement('div');
    div.className = 'form-group';
    div.style.display = 'flex';

    if(options) {
        div.appendChild(createTitle(param, title_text));
        div.appendChild(createSelection(param, options));
    }
    else if(type === 'bool') {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = param;
        input.id = param;
        input.addEventListener('change', function() {input.value = input.checked?true:false;});
        input.checked = def===true;
        input.value = def===true;

        div.appendChild(createTitle(param, title_text, width='210px'));
        div.appendChild(input);
    }
    else if(type === 'url') {
        const urlInput = document.createElement('input');
        urlInput.type = 'text';
        urlInput.name = param;
        urlInput.className = 'form-control';
        urlInput.placeholder = 'Enter a URL or file path';
        urlInput.required = true;
        urlInput.value = def !== 'None' ? def : '';

        // Create browse button
        const browseButton = document.createElement('button');
        browseButton.type = 'button';
        browseButton.className = 'btn btn-secondary';
        browseButton.innerHTML = '<i class="bi bi-folder"></i>';
        browseButton.style.marginLeft = '10px';

        let fileInput = document.getElementById('hiddenFileInput');
        if (!fileInput) {
            fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.id = 'hiddenFileInput';
            fileInput.style.display = 'none';
            document.body.appendChild(fileInput); // Append it to the body
        }
        browseButton.addEventListener('click', () => {fileInput.click();});
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);

                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => console.log('Success:', data))
                .catch(error => console.error('Error:', error));
            }
        });
        div.appendChild(createTitle(param, title_text));
        div.appendChild(urlInput);
        div.appendChild(browseButton);
    }
    else {
        const input = document.createElement('input');
        input.className = 'form-control';
        input.name = param;
        if (def != 'None') input.value = def;

        switch(type) {
            case 'str': input.type = 'text';break;
            case 'int': input.type = 'number'; input.step = '1'; break;
            case 'float': input.type = 'number'; input.step = 'any'; break;
            default: input.type = 'text';
        }
        input.required = true;
        input.addEventListener('input', checkFormCompletion);
        div.appendChild(createTitle(param, title_text));
        div.appendChild(input);
    }

    return div;
}

function showDescriptionModal(button) {
    const description = button.getAttribute("data-description");
    const name = button.getAttribute("data-name");
    document.getElementById("descriptionModalLabel").innerText = name;
    document.getElementById("descriptionModalBody").innerText = description;
    const modal = new bootstrap.Modal(document.getElementById("descriptionModal"));
    modal.show();
}

function showLoadingModal() {
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    setTimeout(() => {loadingModal.show();}, 200);
}

if (baseTask) updateDatasetParameters();
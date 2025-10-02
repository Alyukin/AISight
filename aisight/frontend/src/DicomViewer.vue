<script setup>
import { ref } from 'vue'

// Состояние для файла, сообщений и индикатора загрузки
const file = ref(null)
const responseMessage = ref('')
const fileUrl = ref('')
const isProcessing = ref(false)  // Состояние для отслеживания процесса обработки

// Функция обработки выбора файла
function handleFileChange(event) {
  const selectedFile = event.target.files[0]
  if (selectedFile && selectedFile.name.endsWith('.zip')) {
    file.value = selectedFile
    console.log('Загружен файл:', file.value)
    uploadFile(file.value) // Отправляем файл на сервер
  } else {
    alert('Пожалуйста, выберите ZIP файл')
  }
}

async function uploadFile(selectedFile) {
  const formData = new FormData()
  formData.append('zip_file', selectedFile)
  isProcessing.value = true

  try {
    // Отправка на FastAPI
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('Ошибка на сервере')
    }

    const data = await response.json()

    if (data.error) {
      responseMessage.value = `Ошибка: ${data.error}`
    } else {
      responseMessage.value = 'Файл успешно обработан!'
      fileUrl.value = `http://localhost:8000${data.file_path}` // Ссылка на обработанный файл
    }
  } catch (error) {
    responseMessage.value = `Ошибка: ${error.message}`
  } finally {
    isProcessing.value = false
  }
}
</script>

<template>
  <div id="app">
    <label class="dicom-viewer p-6 rounded-lg shadow-lg flex flex-col items-center justify-center">
      <img src="/upload-icon.svg" alt="Upload" class="upload-icon" />

      <div
        class="cursor-pointer bg-[rgba(255,255,255,0.65)] rounded-lg shadow-lg mx-auto text-center px-16 py-10"
      >
        <span class="text-gray-600 text-[20px]">
          Загрузите ZIP файл с DICOM-снимками<br />или перетащите сюда
        </span>

        <input type="file" class="hidden" @change="handleFileChange" accept=".zip" />

        <div v-if="file" class="mt-4 text-center text-gray-700">Файл выбран: {{ file.name }}</div>
      </div>
    </label>
    

    <div v-if="responseMessage">
      <p class="text-gray-800 text-[18px]">{{ responseMessage }}</p>
      <div
        class="dicom-export mt-6 mx-30 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300"
      >
        <div v-if="isProcessing" class="text-center">
          <p class="text-gray-800 inter__bold text-[20px]">Обработка файла...</p>
        </div>

        <div v-if="!isProcessing && fileUrl" class="flex flex-col justify-center items-center">
          <a
            :href="fileUrl"
            download="results.xlsx"
            class="text-gray-800 inter__bold text-[25px]"
            >Скачать результат</a
          >
        </div>
      </div>
    </div>

  </div>
</template>

import React, { useState, useEffect } from 'react';

// COMPONENTE FIGLIO - Riceve dati tramite props
function TaskItem({ task, onDelete, onToggle }) {
  return (
    <div className="flex items-center justify-between p-3 mb-2 bg-white rounded-lg shadow">
      <div className="flex items-center gap-3">
        <input
          type="checkbox"
          checked={task.completed}
          onChange={() => onToggle(task.id)}
          className="w-5 h-5 cursor-pointer"
        />
        <span className={task.completed ? 'line-through text-gray-400' : 'text-gray-800'}>
          {task.text}
        </span>
      </div>
      <button
        onClick={() => onDelete(task.id)}
        className="px-3 py-1 text-white bg-red-500 rounded hover:bg-red-600"
      >
        Elimina
      </button>
    </div>
  );
}

// COMPONENTE PRINCIPALE
export default function App() {
  // 1. STATE - Dati che cambiano nel tempo
  const [tasks, setTasks] = useState([
    { id: 1, text: 'Studiare React', completed: false },
    { id: 2, text: 'Creare un progetto', completed: false }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [filter, setFilter] = useState('all'); // all, active, completed

  // 2. USEEFFECT - Esegue codice quando il componente si monta o quando cambiano dipendenze
  useEffect(() => {
    console.log(`Hai ${tasks.length} task totali`);
  }, [tasks]); // Si esegue ogni volta che tasks cambia

  // 3. FUNZIONI - Gestione della logica
  const addTask = () => {
    if (inputValue.trim() === '') return;
    
    const newTask = {
      id: Date.now(),
      text: inputValue,
      completed: false
    };
    
    setTasks([...tasks, newTask]); // Crea un nuovo array con il nuovo task
    setInputValue(''); // Reset input
  };

  const deleteTask = (id) => {
    setTasks(tasks.filter(task => task.id !== id));
  };

  const toggleTask = (id) => {
    setTasks(tasks.map(task =>
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  // 4. COMPUTED VALUES - Valori derivati dallo state
  const filteredTasks = tasks.filter(task => {
    if (filter === 'active') return !task.completed;
    if (filter === 'completed') return task.completed;
    return true; // all
  });

  const stats = {
    total: tasks.length,
    active: tasks.filter(t => !t.completed).length,
    completed: tasks.filter(t => t.completed).length
  };

  // 5. RENDER - Ciò che viene visualizzato
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <h1 className="text-4xl font-bold text-center mb-2 text-indigo-900">
          📝 Task Manager React
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Applicazione dimostrativa per imparare React
        </p>

        {/* Statistiche */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-white p-4 rounded-lg shadow text-center">
            <div className="text-2xl font-bold text-indigo-600">{stats.total}</div>
            <div className="text-sm text-gray-600">Totali</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow text-center">
            <div className="text-2xl font-bold text-green-600">{stats.completed}</div>
            <div className="text-sm text-gray-600">Completati</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow text-center">
            <div className="text-2xl font-bold text-orange-600">{stats.active}</div>
            <div className="text-sm text-gray-600">Da fare</div>
          </div>
        </div>

        {/* Input per aggiungere task */}
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addTask()}
              placeholder="Aggiungi un nuovo task..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              onClick={addTask}
              className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-semibold"
            >
              Aggiungi
            </button>
          </div>
        </div>

        {/* Filtri */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded-lg font-medium ${
              filter === 'all' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700'
            }`}
          >
            Tutti
          </button>
          <button
            onClick={() => setFilter('active')}
            className={`px-4 py-2 rounded-lg font-medium ${
              filter === 'active' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700'
            }`}
          >
            Attivi
          </button>
          <button
            onClick={() => setFilter('completed')}
            className={`px-4 py-2 rounded-lg font-medium ${
              filter === 'completed' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700'
            }`}
          >
            Completati
          </button>
        </div>

        {/* Lista task */}
        <div>
          {filteredTasks.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {filter === 'all' ? 'Nessun task. Aggiungine uno!' : `Nessun task ${filter === 'active' ? 'attivo' : 'completato'}`}
            </div>
          ) : (
            filteredTasks.map(task => (
              <TaskItem
                key={task.id}
                task={task}
                onDelete={deleteTask}
                onToggle={toggleTask}
              />
            ))
          )}
        </div>

        {/* Sezione didattica */}
        <div className="mt-8 bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4 text-indigo-900">🎓 Concetti React in questa app:</h2>
          <ul className="space-y-2 text-gray-700">
            <li><strong>useState:</strong> Gestisce lo stato (tasks, inputValue, filter)</li>
            <li><strong>useEffect:</strong> Monitora i cambiamenti dei task</li>
            <li><strong>Props:</strong> TaskItem riceve dati dal componente padre</li>
            <li><strong>Eventi:</strong> onClick, onChange, onKeyPress</li>
            <li><strong>Rendering condizionale:</strong> Mostra messaggi diversi se non ci sono task</li>
            <li><strong>Liste:</strong> map() per renderizzare array di componenti</li>
            <li><strong>Immutabilità:</strong> Usiamo spread operator e filter/map per aggiornare lo state</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
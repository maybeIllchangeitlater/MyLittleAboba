#include "mainwindow.h"

#include <QSignalMapper>
#include <QTimer>

#include "aboba.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(s21::SettingsController& s_controller,
                       s21::MLPController& w_controller, QWidget* parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      settings_controller_(s_controller),
      work_controller_(w_controller) {
  ui->setupUi(this);
  draw_ = new DrawingWindow();
  ui->DrawLayout->addWidget(draw_);
  connect(ui->Test, &QPushButton::clicked, this, &MainWindow::TestPerceptron);
  connect(ui->SelectMLP, &QPushButton::clicked, this,
          &MainWindow::ListAvaliableMLPs);
  ConnectsToLambads();
}

MainWindow::~MainWindow() {
  if (mlp_settings_) delete mlp_settings_;
  if (schedule_) delete schedule_;
  delete draw_;
  delete ui;
}

void MainWindow::ListAvaliableMLPs() {
  std::vector<std::string> info = work_controller_.GetMLPsInfo();
  menu_.clear();

  for (const auto& s : info) menu_.addAction(s.c_str());

  disconnect(&menu_, &QMenu::triggered, nullptr, nullptr);
  disconnect(ui->SelectMLP, &QPushButton::clicked, nullptr, nullptr);

  connect(&menu_, &QMenu::triggered, this, [this, info](QAction* action) {
    HandleMLPSelect(action, info);
    QTimer::singleShot(0, &menu_, &QMenu::close);
  });

  connect(ui->SelectMLP, &QPushButton::clicked, [&]() {
    std::vector<std::string> info = work_controller_.GetMLPsInfo();
    menu_.clear();

    for (const auto& s : info) menu_.addAction(s.c_str());
    menu_.exec(ui->SelectMLP->mapToGlobal(QPoint(0, ui->SelectMLP->height())));
  });
}

void MainWindow::HandleMLPSelect(QAction* action,
                                 std::vector<std::string> info) {
  if (action) {
    QString text = action->text();

    for (size_t i = 0; i < info.size(); ++i) {
      if (info[i].c_str() == text) {
        active_mlp_index_ = i;
        break;
      }
    }
  }
}

void MainWindow::TestPerceptron() {
  if (settings_controller_.PerceptronsAmount()) {
    ui->ErrorWindow->clear();
    auto stats = work_controller_.Test(active_mlp_index_,
                                       ui->TestSample->text().toDouble());
    ui->Accuracy->setText(QString::number(stats[0]));
    ui->Precision->setText(QString::number(stats[1]));
    ui->Recall->setText(QString::number(stats[2]));
    ui->F1measure->setText(QString::number(stats[3]));
    ui->RMS->setText(QString::number(stats[4]));
    ui->Runtime->setText(QString::number(stats[5]));

  } else {
    ui->ErrorWindow->setText("Specify at least one Perceptron");
  }
}

void MainWindow::ConnectsToLambads() {
  connect(ui->Clear, &QPushButton::clicked, this,
          [&](bool) { draw_->ClearScreen(); });
  connect(ui->Train, &QPushButton::clicked, this, [&](bool) {
    if (settings_controller_.PerceptronsAmount()) {
      ui->ErrorWindow->clear();
      work_controller_.Train();
    } else {
      ui->ErrorWindow->setText("Specify at least one Perceptron");
    }
  });
  connect(ui->Save, &QPushButton::clicked, this, [&](bool) {
    if (settings_controller_.PerceptronsAmount() > active_mlp_index_) {
      ui->ErrorWindow->clear();
      work_controller_.Save(active_mlp_index_);
    } else {
      ui->ErrorWindow->setText("Invalid Perceptron index");
    }
  });
  connect(ui->Log, &QPushButton::clicked, this, [&](bool) {
    auto& path = settings_controller_.GetLogPath();
    system(("open " + path + "log.txt").c_str());
  });
  connect(ui->Guess, &QPushButton::clicked, this, [&](bool) {
    if (settings_controller_.PerceptronsAmount()) {
      ui->ErrorWindow->clear();
      char predicted =
          'a' + work_controller_.Predict(draw_->GetPixmap(), active_mlp_index_);
      ui->PredictedLabel->setText(QString(predicted));
    } else {
      ui->ErrorWindow->setText("Specify at least one Perceptron");
    }
  });
  connect(ui->CustomizeMLP, &QPushButton::clicked, this, [&](bool) {
    if (mlp_settings_) delete mlp_settings_;
    mlp_settings_ = new aboba(work_controller_, nullptr);
    mlp_settings_->show();
  });
  connect(ui->CustomizeSchedule, &QPushButton::clicked, this, [&](bool) {
    if (schedule_) delete schedule_;
    schedule_ =
        new TrainingSchedule(settings_controller_, work_controller_, nullptr);
    schedule_->show();
  });
  connect(ui->LoadTest, &QPushButton::clicked, this, [&](bool) {
    auto filename = GetFilename();
    try {
      ui->ErrorWindow->clear();
      work_controller_.LoadTestsData(filename.toStdString().c_str());
    } catch (const std::exception& e) {
      ui->ErrorWindow->setText(QString(e.what()));
    }
  });
  connect(ui->LoadTrain, &QPushButton::clicked, this, [&](bool) {
    auto filename = GetFilename();
    try {
      ui->ErrorWindow->clear();
      work_controller_.LoadTrainData(filename.toStdString().c_str());
    } catch (const std::exception& e) {
      ui->ErrorWindow->setText(QString(e.what()));
    }
  });
}

QString MainWindow::GetFilename() {
  QString filename(QFileDialog::getOpenFileName(
      this, "Open File",
      static_cast<QDir>(QDir::homePath()).absolutePath() + "/Desktop/",
      "Comma-Separated Values (*.csv)"));
  return filename;
}

// fix logging and saving when running multimlp setups

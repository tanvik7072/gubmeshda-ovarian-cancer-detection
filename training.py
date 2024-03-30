from torchvision import models
# TODO: Build and train your network
model = models.vgg16(pretrained = True) #Loading pre-trained network

#Freeze parameters to avoid backpropagation
for param in model.parameters():
    param.requires_grad = False


model = model.to('cuda')

#Defining new untrained feed-forward network
classifier = nn.Sequential(nn.Linear(25088,4096),
                          nn.ReLU(), #activation function - ReLU is effective, computationally inexpensive
                                     #and removes the vanishing gradient problem
                          nn.Dropout(0.2),
                          nn.Linear(4096, 256),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(256, 64),
                          nn.Dropout(0.2), #Removed 20% of data each time, good place to start
                          nn.Linear(64, 6),  #Must be 6 because 6 = number of classes
                          nn.LogSoftmax(dim=1))


classifier = classifier.to('cuda')
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.01)

#Training
epochs = 5 #maybe change to 10 later on? depending on output
train_loss = 0

for epoch in range(epochs):
    #model.train()
    for inputs, labels in train_DL:
      model.train()
      inputs, labels = inputs.to('cuda'), labels.to('cuda')
      optimizer.zero_grad()
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()


    model.eval()
    with torch.inference_mode():
      validation_loss = 0
      accuracy = 0
      for inputs, labels in validation_DL:
          inputs, labels = inputs.to('cuda'), labels.to('cuda')
          outputs = model.forward(inputs)
          running_valid_loss = criterion(outputs, labels).item()
          validation_loss += running_valid_loss

          ps = torch.exp(outputs)
          top_p, top_class = ps.topk(1, dim = 1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Epoch {epoch+1}/{epochs}...",
          f"Train loss: {train_loss/len(train_DL):.3f}..."
        f"Validation loss: {validation_loss/len(validation_DL):.3f}..."
        f"Validation accuracy: {accuracy:.3f}...")
    train_loss = 0

